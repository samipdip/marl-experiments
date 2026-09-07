"""g99 BeliefTrader joiner: one process per trader slot.

Hand-drives a real BeliefTrader (imported from Trading_agents, loaded with the
g99 final checkpoint) against the live arena. Belief assigned externally:
v_i = F + b_i with b_i ~ U{bias_min..bias_max} drawn from the scenario seed.
Waits for a live book before the first cycle (MMs provide the liquidity).
"""

from __future__ import annotations

import argparse
import os
import random
import signal
import sys
import time

import torch

from joiners import common

common.ensure_trading_agents_path()

from Agents_phase2.RL_Agents.ABIDES_MARL.belief_trader import BeliefTrader

from joiners.obs_mapper import map_md


def build_agent(slot: int, cfg: dict) -> BeliefTrader:
    return BeliefTrader(
        name=f"belief_{slot}",
        policy="linear",
        max_order=500,
        initial_cash=20000.0,
        q_max=int(cfg["q_max"]),
        arrival_lo=int(cfg["arrival_lo"]),
        arrival_hi=int(cfg["arrival_hi"]),
        n_steps=240,
        sim_duration=60000.0,
        mode="eval",
        total_timesteps=0,
        eval_freq=0,
        checkpoint_freq=0,
    )


class BeliefJoiner:
    def __init__(self, agent, conn, market, asset_id: int, raw_per_price: int,
                 qty_per_unit: int = 1, aggression: float = 5.0,
                 ref_price: float = 1000.0, ack_logger=None, cycle_offset: int = 0):
        self.agent = agent
        self.conn = conn
        self.market = market
        self.asset_id = asset_id
        self.raw_per_price = raw_per_price
        self.qty_per_unit = qty_per_unit
        self.aggression = aggression
        self.ref_price = ref_price
        self.ack_logger = ack_logger
        self.cycle_offset = int(cycle_offset)
        self._last_two_sided_md = None  # last valid two-sided book (§5g fix)

    def _observation_md(self, md: dict | None) -> tuple[dict | None, int]:
        """md used for the decision; substitute on one-sided books (§5g).

        Same rationale as MMJoiner._observation_md: a one-sided book would
        zero the belief trader's touch (_signed_edge falls back to ref +- tick),
        injecting ref-biased observations the policy never saw in training.
        Without the touch context, the trade gate stays on real md anyway.
        Returns (md_for_observation, obs_ffill flag).
        """
        if md is not None and md.get("bid", 0) > 0 and md.get("ask", 0) > 0:
            self._last_two_sided_md = md
            return md, 0
        if self._last_two_sided_md is not None:
            return self._last_two_sided_md, 1
        return md, 1  # no two-sided book seen yet: nothing better to anchor on

    def run_cycle(self, cycle: int) -> dict:
        agent = self.agent
        n_fills = 0
        fill_vol = 0
        for f in self.conn.drain_fills():
            d = common.fill_to_dict(f)
            d["price"] = common.to_display(d["price"], self.raw_per_price)
            d["volume"] = int(round(d["volume"] / self.qty_per_unit))
            agent.on_fill(d)
            n_fills += 1
            fill_vol += d["volume"]
        n_acks = 0
        n_rejects = 0
        n_cash_rejects = 0
        last_reject = ""
        for a in self.conn.drain_acks():
            d = common.ack_to_dict(a)
            n_acks += 1
            rn = common.reject_name(d)
            if rn:
                n_rejects += 1
                last_reject = rn
                if rn == "InsufficientCash":
                    n_cash_rejects += 1
            if self.ack_logger is not None:
                self.ack_logger.row(
                    cycle=cycle, tag=d.get("client_tag"), side=d.get("side"),
                    status=d.get("status"), reject=rn or 0,
                    price=common.to_display(d.get("price") or 0, self.raw_per_price),
                    remaining=d.get("remaining"),
                )
        raw_md = self.market.get_latest(self.asset_id, attempts=3)
        real_md = map_md(raw_md, self.raw_per_price, self.qty_per_unit)
        md, obs_ffill = self._observation_md(real_md)
        # touch state only updated from valid two-sided books (§5g fix):
        # a 0 touch would corrupt _signed_edge (ref fallback) and persist
        if real_md:
            if real_md["bid"] > 0:
                agent._last_best_bid = real_md["bid"]
            if real_md["ask"] > 0:
                agent._last_best_ask = real_md["ask"]

        sim_t = float(cycle + 1)
        obs = agent.build_observation(sim_t, md)
        action = agent.sample_action(obs, deterministic=True)
        orders = agent.action_to_orders(action, sim_t, md)
        sent = common.submit_orders(self.conn, orders, self.asset_id,
                                    self.raw_per_price, self.qty_per_unit,
                                    md=real_md, aggression=self.aggression,
                                    ref_price=self.ref_price)
        agent._decision_step += 1
        agent._step_fills = []

        mid = (real_md["bid"] + real_md["ask"]) / 2.0 \
            if real_md and real_md["bid"] and real_md["ask"] else 0.0
        qty = sent[-1]["qty"] if sent else 0
        side = sent[-1]["side"] if sent else -1
        return {
            "cycle": cycle, "gcycle": self.cycle_offset + cycle,
            "ts": time.time(), "bid": real_md["bid"] if real_md else 0.0,
            "ask": real_md["ask"] if real_md else 0.0, "mid": mid, "v": agent._v,
            "active": int(sim_t >= agent._active_from),
            "beta": float(action), "qty": qty, "side": side,
            "n_fills": n_fills, "fill_vol": fill_vol,
            "inventory": agent._inventory, "cash": agent._cash,
            "obs_norm": float(torch.norm(torch.FloatTensor(obs)).item()),
            "md_ok": int(bool(getattr(self.market, "ok", False))),
            "md_src": self.market.src,
            "n_acks": n_acks, "n_rejects": n_rejects,
            "n_cash_rejects": n_cash_rejects, "last_reject": last_reject,
            "obs_ffill": obs_ffill,
        }


LOG_FIELDS = ["cycle", "gcycle", "ts", "bid", "ask", "mid", "v", "active",
              "beta", "qty", "side", "n_fills", "fill_vol", "inventory", "cash",
              "obs_norm", "md_ok", "md_src", "n_acks", "n_rejects",
              "n_cash_rejects", "last_reject", "obs_ffill"]


def main() -> None:
    ap = argparse.ArgumentParser(description="g99 BeliefTrader arena joiner")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--config", default="config/scenarios.yaml")
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=None,
                    help="override session rounds (churn: trader lifetime)")
    ap.add_argument("--spawn-id", type=int, default=None,
                    help="unique spawn id (churn): seeds RNG, labels logs")
    ap.add_argument("--bias", type=int, default=None,
                    help="override the bias draw (churn: drawn by the orchestrator)")
    ap.add_argument("--cycle-offset", type=int, default=0,
                    help="global session cycle this spawn starts at (churn); "
                         "logged as gcycle for alignment. Agent sim_t stays local.")
    args = ap.parse_args()

    cfg = common.load_scenario(args.scenario, args.config)
    state = common.read_state(args.state)
    raw_per_price = int(cfg["raw_per_price"])
    qty_per_unit = int(cfg.get("qty_per_unit", 1))

    spawn_id = args.spawn_id if args.spawn_id is not None else args.slot
    label = f"belief_s{spawn_id}" if args.spawn_id is not None else f"belief_{args.slot}"
    seed = common.seed_for(cfg["seed"], "belief", spawn_id)
    random.seed(seed)
    torch.manual_seed(seed)

    agent = build_agent(args.slot, cfg)
    # Churn reuses physical slots; distinct trained policies are 0..n_traders-1
    # (e.g. matched roster: 3 policies over 4 churn slots).
    ckpt_idx = args.slot % max(int(cfg["n_traders"]), 1)
    ckpt = common.final_checkpoint(cfg["ckpt_root"], "belief", "belief", ckpt_idx)
    agent.load(ckpt)
    b_i = args.bias if args.bias is not None else random.randint(int(cfg["bias_min"]), int(cfg["bias_max"]))
    agent.set_belief(float(cfg["f"]) + b_i)
    print(f"[{label}] loaded {ckpt}  v={agent._v} (F={cfg['f']}, b={b_i})")

    platform = common.connect_platform()
    arena = platform.reconnect_arena(state["arena_id"])
    trader_idx = int(cfg["n_mms"]) + args.slot
    conn = arena.connect_trader(trader_idx, session_tag=8000 + args.slot)
    asset_id = state["asset_id"]
    conn.activate(asset_id)
    market = common.MarketSource(platform)
    print(f"[belief_{args.slot}] md status: {market.status()}")

    md0_raw = common.wait_for_book(market, asset_id,
                                   timeout_secs=min(60.0, float(state["rounds"])))
    md0 = map_md(md0_raw, raw_per_price, qty_per_unit)
    agent.on_episode_start(md0)

    logger = common.CycleLogger(args.out, label, LOG_FIELDS)
    ack_logger = common.CycleLogger(
        args.out, f"{label}_ack",
        ["cycle", "tag", "side", "status", "reject", "price", "remaining"])
    joiner = BeliefJoiner(agent, conn, market, asset_id, raw_per_price,
                          qty_per_unit, float(cfg.get("market_aggression", 5.0)),
                          float(cfg["ref_price"]), ack_logger,
                          cycle_offset=args.cycle_offset)
    rounds = int(args.rounds) if args.rounds else int(state["rounds"])
    cycle_secs = float(cfg["cycle_secs"])
    rng = random.Random(seed + 13)

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    t0 = float(state.get("t_start", 0.0)) or time.time() + 5.0
    t0 = max(t0, time.time())
    cycle = 0
    total_rejects = 0
    try:
        while running and cycle < rounds:
            t_next = t0 + (cycle + 1) * cycle_secs + rng.uniform(0.0, float(cfg["jitter_secs"]))
            delay = t_next - time.time()
            if delay > 0:
                time.sleep(delay)
            row = joiner.run_cycle(cycle)
            total_rejects += row["n_rejects"]
            logger.row(**row)
            cycle += 1
    finally:
        try:
            conn.cancel_all(asset_id)
            conn.drain_acks()
            time.sleep(0.2)
        except Exception as e:
            print(f"[{label}] cleanup warning: {e}")
        logger.close()
        ack_logger.close()
        print(f"[{label}] done after {cycle} cycles  "
              f"inv={agent._inventory} cash={agent._cash:.2f} "
              f"rejects={total_rejects}")


if __name__ == "__main__":
    main()
