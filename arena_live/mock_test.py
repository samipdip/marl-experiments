"""Offline adapter tests for arena_live joiners (rung 0 — no exchange needed).

Runs the real sandbox agents (PPOHjbMarketMaker, BeliefTrader) inside the
joiner drivers against FakeConn/FakeMarket, and checks the mechanics:
obs shapes, deterministic kappa range, order translation + raw-price/qty
scaling, cancel-before-repost, fill accounting at display scale, belief q_max
caps, arrival-jitter uniformity.

Exchange scale: raw price = display x 100 (tick_size 100); raw qty = units
x 100 (lot_size 100). Agents live in display/unit scale.

Checkpoints are loaded when present (QFRCAI); otherwise the randomly
initialized policy is used — the mechanics under test don't depend on weights.

Usage:  python mock_test.py            (from arena_live/, torch + Trading_agents required)
"""

from __future__ import annotations

import os
import random

from joiners import common

common.ensure_trading_agents_path()

from joiners.mm_joiner import MMJoiner, build_agent as build_mm
from joiners.belief_joiner import BeliefJoiner, build_agent as build_belief
from joiners.obs_mapper import map_md

import torch

CFG = {
    "ckpt_root": os.path.expanduser("~/checkpoints/beliefs_pop_240_g99"),
    "q_max": 20, "ref_price": 1000.0, "raw_per_price": 10000, "qty_per_unit": 100,
    "f": 1003.0, "bias_min": -3, "bias_max": 3, "arrival_lo": 0, "arrival_hi": 60,
    "seed": 42,
}
R = 10000  # raw units per display price unit (from_float: display x 10^4)
QPU = 100  # raw shares per unit qty (lot_size 100)


class FakeConn:
    def __init__(self):
        self.calls = []
        self.fills_q = []
        self.acks_q = []

    def submit_limit(self, asset_id, side, price, qty, tag):
        self.calls.append(("limit", side, price, qty, tag))

    def submit_market(self, asset_id, side, qty, tag):
        self.calls.append(("market", side, 0, qty, tag))

    def submit_ioc(self, asset_id, side, delta, qty, tag):
        self.calls.append(("ioc", side, 0, qty, tag))

    def cancel_by_tag(self, tag, asset_id, side):
        self.calls.append(("cancel", side, 0, 0, tag))

    def cancel_all(self, asset_id, side=None):
        self.calls.append(("cancel_all", -1, 0, 0, 0))

    def drain_fills(self):
        q, self.fills_q = self.fills_q, []
        return q

    def drain_acks(self):
        q, self.acks_q = self.acks_q, []
        return q

    def activate(self, asset_id):
        pass


class FakeMarket:
    src = "fake"
    ok = True

    def __init__(self):
        self.md = None

    def set(self, bid_raw, ask_raw, vbuy=0, vsell=0, levels=5, qty_units=3):
        bids = [(bid_raw - i * 100, qty_units * QPU) for i in range(levels)]
        asks = [(ask_raw + i * 100, qty_units * QPU) for i in range(levels)]
        self.md = {"bid": bid_raw, "ask": ask_raw,
                   "bid_qty": qty_units * QPU, "ask_qty": qty_units * QPU,
                   "bids": bids, "asks": asks,
                   "volume_buy": vbuy * QPU, "volume_sell": vsell * QPU}

    def get_latest(self, asset_id, attempts=1, retry_secs=0.0):
        return self.md


def make_fill(side, price_raw, volume_units, tag=1):
    return {"order_id": 1, "asset_id": 1, "price": price_raw,
            "volume": volume_units * QPU,
            "remaining": 0, "side": side, "client_tag": tag, "ts": 0,
            "is_maker": True, "fee_direction": 0, "status": 5, "fill_fee": 0}


def load_if_present(agent, name):
    path = os.path.join(CFG["ckpt_root"], name)
    if os.path.exists(path):
        agent.load(path)
        print(f"  loaded ckpt: {name}")
    else:
        print(f"  no ckpt ({name}); using random init")


def test_obs_mapper():
    md = map_md({"bid": 9_995_000, "ask": 10_005_000,
                 "bid_qty": 3 * QPU, "ask_qty": 3 * QPU,
                 "bids": [(9_995_000, 3 * QPU), (9_985_000, 2 * QPU)],
                 "asks": [(10_005_000, 4 * QPU)],
                 "volume_buy": 17 * QPU, "volume_sell": 9 * QPU}, R, QPU)
    assert md["bid"] == 999.5 and md["ask"] == 1000.5
    assert md["bids"][0] == (999.5, 3.0) and md["asks"][0][1] == 4.0
    assert md["bid_qty"] == 3.0 and md["ask_qty"] == 3.0
    assert md["taker_volume_buy"] == 17 and md["taker_volume_sell"] == 9
    assert map_md(None, R, QPU) is None
    print("PASS obs_mapper")


def test_mm_joiner():
    agent = build_mm(0, CFG)
    assert agent.model.shared[0].in_features == 9, "MM obs_dim must be 9"
    assert agent.model.action_dim == 2, "MM action_dim must be 2"
    load_if_present(agent, "ppo_0/ppo_hjb_mm_0_step_180029_final.pt")

    conn, market = FakeConn(), FakeMarket()
    market.set(9_995_000, 10_005_000, vbuy=10, vsell=12)
    agent.on_episode_start(map_md(market.get_latest(1), R, QPU))
    j = MMJoiner(agent, conn, market, 1, R, QPU)

    for c in range(5):
        market.set(9_995_000 - c * 10_000, 10_005_000 + c * 10_000, vbuy=10 + 5 * c, vsell=12 + 4 * c)
        row = j.run_cycle(c)
        assert row["mid"] == 1000.0
        assert 0.01 <= row["kappa_bid"] <= 2.0 and 0.01 <= row["kappa_ask"] <= 2.0
    limits = [x for x in conn.calls if x[0] == "limit"]
    cancels = [x for x in conn.calls if x[0] == "cancel"]
    assert len(limits) >= 8, "2-sided quotes expected every cycle"
    assert len(cancels) >= 4, "cancel-before-repost expected"
    for _, side, price_raw, qty_raw, tag in limits:
        assert price_raw % 100 == 0 and 9_900_000 < price_raw < 10_100_000, \
            f"raw price scale {price_raw}"
        assert qty_raw == QPU, f"raw qty must be 1 unit = {QPU}, got {qty_raw}"

    conn.fills_q = [make_fill(0, 10_005_000, 2)]
    row = j.run_cycle(5)
    assert agent._inventory == 2 and abs(agent._cash + 2 * 1000.5) < 1e-6
    assert row["fill_vol"] == 2
    print("PASS mm_joiner (orders, scaling, cancels, fill accounting)")


def test_mm_warmup_anchor():
    agent = build_mm(1, CFG)
    load_if_present(agent, "ppo_1/ppo_hjb_mm_1_step_180029_final.pt")

    conn, market = FakeConn(), FakeMarket()
    market.md = None  # empty book at cold start
    agent.on_episode_start(None)
    j = MMJoiner(agent, conn, market, 1, R, QPU,
                 ref_price=1000.0, ref_anchor_warmup=True)

    row = j.run_cycle(0)
    assert row["mid"] == 1000.0, f"warmup must anchor at ref, got {row['mid']}"
    limits0 = [x for x in conn.calls if x[0] == "limit"]
    assert limits0, "warmup cycle must still quote 2-sided"
    for _, side, price_raw, qty_raw, tag in limits0:
        assert 9_980_000 <= price_raw <= 10_020_000, \
            f"warmup quotes must stay near ref (display {price_raw / R:.1f})"

    market.set(9_995_000, 10_005_000, vbuy=5, vsell=5)
    row = j.run_cycle(1)
    assert row["mid"] == 1000.0  # symmetric book still mid 1000
    assert agent._last_best_bid == 999.5 and agent._last_best_ask == 1000.5, \
        "cycle 1 must re-anchor on the observed book, not warmup md"

    j2 = MMJoiner(build_mm(2, CFG), FakeConn(), market, 1, R, QPU)
    row = j2.run_cycle(0)
    assert row["bid"] == 999.5 and row["ask"] == 1000.5, \
        "warmup disabled (default) keeps observed-book behavior"
    print("PASS mm warmup anchor (cycle-0 ref-centered, cycle-1 re-anchor)")


def test_one_sided_md_substitution():
    """§5g fix: one-sided book observations must not feed the policy."""
    # MM joiner
    agent = build_mm(0, CFG)
    load_if_present(agent, "ppo_0/ppo_hjb_mm_0_step_180029_final.pt")
    conn, market = FakeConn(), FakeMarket()
    market.set(9_995_000, 10_005_000, vbuy=5, vsell=5)
    agent.on_episode_start(map_md(market.get_latest(1), R, QPU))
    j = MMJoiner(agent, conn, market, 1, R, QPU)

    row = j.run_cycle(0)
    assert row["obs_ffill"] == 0, "two-sided book -> no substitution"
    # ask side disappears (bid-only book)
    market.set(9_995_000, 0, vbuy=5, vsell=0)
    row = j.run_cycle(1)
    assert row["obs_ffill"] == 1, "one-sided book -> substitution flagged"
    assert agent._current_mid == 1000.0, \
        f"anchor must stay at last two-sided mid 1000, got {agent._current_mid}"
    assert agent._last_best_bid == 999.5 and agent._last_best_ask == 1000.5, \
        "touch state must come from the substituted two-sided md"
    assert row["bid"] == 999.5 and row["ask"] == 0.0, \
        "log must record the REAL (one-sided) book"
    # book returns two-sided: substitution released
    market.set(9_996_000, 10_006_000, vbuy=5, vsell=5)
    row = j.run_cycle(2)
    assert row["obs_ffill"] == 0 and row["mid"] == 1000.1

    # cold start with one-sided book and NO prior two-sided book -> ref fallback
    agent2 = build_mm(1, CFG)
    load_if_present(agent2, "ppo_1/ppo_hjb_mm_1_step_180029_final.pt")
    conn2, market2 = FakeConn(), FakeMarket()
    market2.set(9_995_000, 0, vbuy=5, vsell=0)
    agent2.on_episode_start(None)
    j2 = MMJoiner(agent2, conn2, market2, 1, R, QPU, ref_price=1000.0)
    row = j2.run_cycle(0)
    assert row["obs_ffill"] == 1
    assert agent2._current_mid == 1000.0, "cold start one-sided -> ref fallback"

    # belief joiner: touch state must not be corrupted by a zero side
    agent3 = build_belief(0, CFG)
    load_if_present(agent3, "belief_0/belief_0_step_180029_final.pt")
    agent3.set_belief(1003.0)
    agent3.on_episode_start(None)
    agent3._active_from = 0
    conn3, market3 = FakeConn(), FakeMarket()
    market3.set(9_995_000, 10_005_000, vbuy=5, vsell=5)
    j3 = BeliefJoiner(agent3, conn3, market3, 1, R, QPU)
    j3.run_cycle(0)
    assert agent3._last_best_bid == 999.5 and agent3._last_best_ask == 1000.5
    market3.set(9_995_000, 0, vbuy=5, vsell=0)
    row = j3.run_cycle(1)
    assert row["obs_ffill"] == 1
    assert agent3._last_best_ask == 1000.5, \
        "ask touch must survive an ask-empty book (no zero-corruption)"
    assert agent3._last_best_bid == 999.5
    print("PASS one-sided md substitution (mm + belief, cached + ref fallback)")



def test_belief_joiner():
    agent = build_belief(0, CFG)
    assert agent.model.shared[0].in_features == 25, "trader obs_dim must be 25"
    load_if_present(agent, "belief_0/belief_0_step_180029_final.pt")
    agent.set_belief(1003.0)
    agent.on_episode_start(None)
    agent._active_from = 0

    conn, market = FakeConn(), FakeMarket()
    market.set(9_995_000, 10_005_000, vbuy=5, vsell=5)
    out_dir = "/tmp/opencode/mock_belief_out"
    os.makedirs(out_dir, exist_ok=True)
    ack_logger = common.CycleLogger(
        out_dir, "belief_0_ack",
        ["cycle", "tag", "side", "status", "reject", "price", "remaining"])
    j = BeliefJoiner(agent, conn, market, 1, R, QPU, ack_logger=ack_logger)

    row = j.run_cycle(0)
    edge = 1003.0 - 1000.5
    if row["qty"] > 0:
        assert row["side"] == 0, "v above ask -> buy"
        assert row["qty"] == min(int(round(row["beta"] * edge)), 500, CFG["q_max"])
        markets = [x for x in conn.calls if x[0] == "market"]
        assert markets, "expected a market order (ask present)"
        assert markets[-1][3] == row["qty"] * QPU, "raw qty scaling"
    else:
        assert row["beta"] * edge < 0.5, "no order only if beta*edge rounds to 0"
    assert row["n_rejects"] == 0 and row["n_cash_rejects"] == 0

    # ack parity: rejects are drained, counted, named, and logged to CSV
    conn.acks_q = [
        {"client_tag": 9, "side": 0, "status": 8, "reject_reason": 0,
         "price": 10_005_000, "remaining": 0},
        {"client_tag": 10, "side": 0, "status": 9, "reject_reason": 5,
         "price": 10_005_000, "remaining": 100},
        {"client_tag": 11, "side": 1, "status": 9, "reject_reason": "BookEmpty",
         "price": 0, "remaining": 0},
    ]
    row = j.run_cycle(1)
    assert row["n_acks"] == 3 and row["n_rejects"] == 2
    assert row["n_cash_rejects"] == 1 and row["last_reject"] == "BookEmpty", row
    ack_path = os.path.join(out_dir, "belief_0_ack_cycle_log.csv")
    assert os.path.exists(ack_path), "belief ack CSV must be written"
    import csv as _csv
    ack_rows = list(_csv.DictReader(open(ack_path)))
    assert len(ack_rows) == 3
    assert ack_rows[1]["reject"] == "InsufficientCash"
    ack_logger.close()
    print("PASS belief_joiner acks (drain, reject naming, cash counter, CSV)")

    conn.calls.clear()
    market.md = None  # empty book -> market order must be skipped
    j.run_cycle(4)
    assert not [x for x in conn.calls if x[0] == "market"], "skip when touch missing"
    market.set(9_995_000, 10_005_000)

    agent._inventory = CFG["q_max"]
    agent._cash = 0.0
    market.set(9_995_000, 10_005_000)
    row = j.run_cycle(1)
    assert row["qty"] == 0, "q_max cap must block further buys"

    agent._inventory = -CFG["q_max"]
    agent.set_belief(997.0)
    market.set(10_015_000, 10_025_000)
    row = j.run_cycle(2)
    assert row["qty"] == 0, "q_max cap must block further sells at -q_max"

    conn.fills_q = [make_fill(0, 10_005_000, 2)]
    row = j.run_cycle(3)
    assert agent._inventory == -CFG["q_max"] + 2
    print("PASS belief_joiner (obs, action->order, q_max caps, fill scaling)")


def test_scheduler_uniformity():
    n_agents, n_rounds = 21, 2000
    wins = [0] * n_agents
    rngs = [random.Random(1000 + i) for i in range(n_agents)]
    for _ in range(n_rounds):
        offsets = [rng.uniform(0.0, 0.5) for rng in rngs]
        wins[offsets.index(min(offsets))] += 1
    expected = n_rounds / n_agents
    assert max(wins) < 1.35 * expected and min(wins) > 0.65 * expected, wins
    print(f"PASS scheduler uniformity (rank-1 wins min={min(wins)} max={max(wins)}, "
          f"expected≈{expected:.0f})")


def test_draw_life():
    import random as _r
    from orchestrator import draw_life
    rng = _r.Random(7)
    lives = [draw_life(rng, 60, 20, 150) for _ in range(2000)]
    assert min(lives) >= 20 and max(lives) <= 150
    assert 40 < sum(lives) / len(lives) < 80, "mean should stay near life_mean"
    print(f"PASS draw_life (min={min(lives)} max={max(lives)} mean={sum(lives)/len(lives):.1f})")


def test_paired_biases():
    import random as _r
    from orchestrator import PairedBiasQueue, draw_paired_biases
    rng = _r.Random(11)
    for trial in range(200):
        n = 4 if trial % 2 == 0 else 3
        bs = draw_paired_biases(rng, n, 3)
        assert len(bs) == n
        assert sum(bs) == 0, f"net bias must be 0, got {sum(bs)} for {bs}"
        assert all(b != 0 for b in bs[: 2 * (n // 2)]), "paired magnitudes nonzero"
        assert all(abs(b) <= 3 for b in bs)
    q = PairedBiasQueue(_r.Random(3), 3)
    stream = [q.draw() for _ in range(1000)]
    assert all(sum(stream[i:i + 2]) == 0 for i in range(0, 1000, 2)), \
        "churn queue must emit +/- pairs"
    assert abs(sum(stream)) <= 3, "running net bias stays bounded"
    print("PASS paired biases (fixed net-zero set, churn pair queue)")


def test_final_checkpoint():
    import tempfile
    from joiners.common import final_checkpoint
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "ppo_0")
        os.makedirs(d)
        open(os.path.join(d, "ppo_hjb_mm_0_step_90000_final.pt"), "w").close()
        open(os.path.join(d, "ppo_hjb_mm_0_step_180029_final.pt"), "w").close()
        open(os.path.join(d, "ppo_hjb_mm_0_step_5000.pt"), "w").close()
        got = final_checkpoint(td, "ppo", "ppo_hjb_mm", 0)
        assert got.endswith("step_180029_final.pt"), got
        miss = final_checkpoint(td, "ppo", "ppo_hjb_mm", 3)
        assert "ppo_3" in miss and "180029" in miss, miss
    print("PASS final_checkpoint (newest *_final.pt, fallback path)")


def test_matched_profile():
    cfg = common.load_scenario("matched_calm", "config/scenarios.yaml")
    assert cfg["n_mms"] == 7 and cfg["n_traders"] == 3
    assert cfg["noise_lambda"] == 0.47 and cfg["mm_lambda"] == 0.47
    assert cfg["ckpt_root"].endswith("matched_7x3_g99")
    assert cfg["bias_mode"] == "paired" and cfg["f"] == 1000
    legacy = common.load_scenario("calm", "config/scenarios.yaml")
    assert legacy["n_mms"] == 6 and legacy["noise_lambda"] == 1.0
    assert "beliefs_pop_240_g99" in legacy["ckpt_root"]
    print("PASS matched profile (7+3 lam 0.47; legacy calm unchanged)")


def test_val_scenarios():
    for scen, f in (("val_calm", 1000), ("val_bull", 1003), ("val_bear", 997)):
        cfg = common.load_scenario(scen, "config/scenarios.yaml")
        assert cfg["n_mms"] == 7 and cfg["n_traders"] == 3
        assert cfg["noise_lambda"] == 0.47 and cfg["mm_lambda"] == 0.47
        assert cfg["ckpt_root"].endswith("matched_7x3_g99_125k")
        assert cfg["bias_mode"] == "random" and cfg["f"] == f
    print("PASS val scenarios (matched125k profile, iid random biases)")


def test_val2_scenarios():
    for scen, f in (("val2_calm", 1000), ("val2_bull", 1003), ("val2_bear", 997)):
        cfg = common.load_scenario(scen, "config/scenarios.yaml")
        assert cfg["bias_mode"] == "random" and cfg["f"] == f
        assert cfg["cycle_secs"] == 0.5 and cfg["jitter_secs"] == 0.25
        assert cfg["ckpt_root"].endswith("matched_7x3_g99_125k")
    print("PASS val2 scenarios (0.5s cadence, iid random biases)")


def main():
    torch.manual_seed(0)
    random.seed(0)
    test_obs_mapper()
    test_mm_joiner()
    test_mm_warmup_anchor()
    test_one_sided_md_substitution()
    test_belief_joiner()
    test_scheduler_uniformity()
    test_draw_life()
    test_paired_biases()
    test_final_checkpoint()
    test_matched_profile()
    test_val_scenarios()
    test_val2_scenarios()
    print("ALL MOCK TESTS PASSED")


if __name__ == "__main__":
    main()
