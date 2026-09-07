"""g99 PPO-Hjb-MM joiner: one process per MM slot.

Hand-drives a real PPOHjbMarketMaker (imported from Trading_agents, loaded
with the g99 final checkpoint) against the live arena:

    drain fills -> map live md -> set _last_best_bid/ask -> build_observation
    -> deterministic action (kappa_bid, kappa_ask) -> action_to_orders
    -> cancel-by-tag + repost limits

Cycle cadence: sleep to next 1s boundary + U(0, jitter) so arrival order at
the exchange is a fresh random permutation each round.
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

from Agents_phase2.RL_Agents.ABIDES_MARL.ppo_hjb_market_maker import PPOHjbMarketMaker

from joiners.obs_mapper import map_md


def build_agent(slot: int, cfg: dict) -> PPOHjbMarketMaker:
    lam = float(cfg.get("mm_lambda", 1.0))
    return PPOHjbMarketMaker(
        name=f"ppo_hjb_mm_{slot}",
        lot_size=1,
        ref_price=float(cfg["ref_price"]),
        tick=1.0,
        dt=1.0,
        sim_duration=240.0,
        N=240,
        alpha=0.01,
        phi=0.001,
        lambda_plus=lam,
        lambda_minus=lam,
        kappa_min=0.01,
        kappa_max=2.0,
        kappa_ref=0.8,
        q_min=-int(cfg["q_max"]),
        q_max=int(cfg["q_max"]),
        hjb_dt=0.5,
        lambda_ema_decay=0.3,
        hjb_update_freq=0,
        n_lob_levels=5,
        n_steps=240,
        tag_start=60000 + slot * 100,
        mode="eval",
        total_timesteps=0,
        eval_freq=0,
        checkpoint_freq=0,
    )


class MMJoiner:
    def __init__(self, agent, conn, market, asset_id: int, raw_per_price: int,
                 qty_per_unit: int = 1, ack_logger=None,
                 ref_price: float = 1000.0, ref_anchor_warmup: bool = False):
        self.agent = agent
        self.conn = conn
        self.market = market
        self.asset_id = asset_id
        self.raw_per_price = raw_per_price
        self.qty_per_unit = qty_per_unit
        self.ack_logger = ack_logger
        self.ref_price = float(ref_price)
        self.aggression = 5.0
        self.ref_anchor_warmup = bool(ref_anchor_warmup)
        self._warmup_done = False
        self._last_two_sided_md = None  # last valid two-sided book (§5g fix)

    def _ref_md(self) -> dict:
        """Synthetic ref-centered md for the cold-start decision (cycle 0).

        Without this, the first MM to post sees an empty book and anchors on
        ref with its own (asymmetric) cold-start kappas; every other MM then
        re-anchors on that wide asymmetric book via _get_mid -> session starts
        ~996 instead of ~1000. Feeding all MMs the same ref-centered md for
        exactly one cycle makes the cycle-0 book an aggregate of ref-centered
        quotes, eliminating the first-poster lottery.
        """
        r = self.ref_price
        return {"bid": r - 1.0, "ask": r + 1.0,
                "bid_qty": 1.0, "ask_qty": 1.0,
                "bids": [(r - 1.0, 1.0)], "asks": [(r + 1.0, 1.0)],
                "taker_volume_buy": 0, "taker_volume_sell": 0}

    def _observation_md(self, md: dict | None) -> tuple[dict | None, int]:
        """md used for the decision; substitute on one-sided books (§5g).

        On the live exchange, sweeping flow + async cancel/repost transiently
        empties one book side. _get_mid then returns the surviving side's
        price as "mid" and every observing MM re-anchors on the collapsed
        value (measured: 2.45% of MM observations one-sided; episodic
        collapses account for the whole down-drift, HANDOFF §5g). The
        sandbox never shows one-sided books at decision time, so the policy
        was never trained on this input. Substitute the last valid
        two-sided book for OBSERVATION only; if none exists yet, use the
        ref-centered synthetic book. Submissions still see real conditions.
        Returns (md_for_observation, obs_ffill flag).
        """
        if md is not None and md.get("bid", 0) > 0 and md.get("ask", 0) > 0:
            self._last_two_sided_md = md
            return md, 0
        if self._last_two_sided_md is not None:
            return self._last_two_sided_md, 1
        return self._ref_md(), 1

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
        last_reject = 0
        ack_rows = []
        for a in self.conn.drain_acks():
            d = common.ack_to_dict(a)
            n_acks += 1
            rn = common.reject_name(d)
            if rn:
                n_rejects += 1
                last_reject = rn
            ack_rows.append(d)

        raw_md = self.market.get_latest(self.asset_id, attempts=3)
        real_md = map_md(raw_md, self.raw_per_price, self.qty_per_unit)
        warmup = self.ref_anchor_warmup and not self._warmup_done
        if warmup:
            self._last_two_sided_md = None
            md = self._ref_md()
            self._warmup_done = True
            obs_ffill = 0
        else:
            md, obs_ffill = self._observation_md(real_md)
            if obs_ffill == 0 and md is not None:
                self._last_two_sided_md = md
        if md and not warmup:
            agent._last_best_bid = md["bid"]
            agent._last_best_ask = md["ask"]

        sim_t = float(cycle + 1)
        obs = agent.build_observation(sim_t, md)
        action = agent.sample_action(obs, deterministic=True)
        orders = agent.action_to_orders(action, sim_t, md)
        # submissions still gate on the REAL book (touch exists etc.)
        sent = common.submit_orders(self.conn, orders, self.asset_id,
                                    self.raw_per_price, self.qty_per_unit,
                                    md=real_md, aggression=float(getattr(self, "aggression", 5.0)),
                                    ref_price=float(getattr(self, "ref_price", 1000.0)))
        agent._decision_step += 1
        agent._step_fills = []

        # log the REAL book state; obs_ffill marks substituted observations
        mid = (real_md["bid"] + real_md["ask"]) / 2.0 \
            if real_md and real_md["bid"] and real_md["ask"] else agent._current_mid
        row = {
            "cycle": cycle, "ts": time.time(),
            "bid": real_md["bid"] if real_md else 0.0,
            "ask": real_md["ask"] if real_md else 0.0, "mid": mid,
            "kappa_bid": float(agent._last_kappa_bid),
            "kappa_ask": float(agent._last_kappa_ask),
            "n_orders": len(sent), "n_fills": n_fills, "fill_vol": fill_vol,
            "inventory": agent._inventory, "cash": agent._cash,
            "obs_norm": float(torch.norm(torch.FloatTensor(obs)).item()),
            "md_ok": int(bool(getattr(self.market, "ok", False))),
            "md_src": self.market.src,
            "n_acks": n_acks, "n_rejects": n_rejects, "last_reject": last_reject,
            "obs_ffill": obs_ffill,
        }
        for d in ack_rows:
            self.ack_logger.row(
                cycle=cycle, tag=d.get("client_tag"), side=d.get("side"),
                status=d.get("status"), reject=common.reject_name(d) or 0,
                price=common.to_display(d.get("price") or 0, self.raw_per_price),
                remaining=d.get("remaining"),
            )
        return row


LOG_FIELDS = ["cycle", "ts", "bid", "ask", "mid", "kappa_bid", "kappa_ask",
              "n_orders", "n_fills", "fill_vol", "inventory", "cash", "obs_norm",
              "md_ok", "md_src", "n_acks", "n_rejects", "last_reject",
              "obs_ffill"]


def main() -> None:
    ap = argparse.ArgumentParser(description="g99 PPO-Hjb-MM arena joiner")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--config", default="config/scenarios.yaml")
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = common.load_scenario(args.scenario, args.config)
    state = common.read_state(args.state)
    raw_per_price = int(cfg["raw_per_price"])
    qty_per_unit = int(cfg.get("qty_per_unit", 1))

    random.seed(common.seed_for(cfg["seed"], "mm", args.slot))
    torch.manual_seed(common.seed_for(cfg["seed"], "mm", args.slot))

    agent = build_agent(args.slot, cfg)
    ckpt = common.final_checkpoint(cfg["ckpt_root"], "ppo", "ppo_hjb_mm",
                                   args.slot)
    agent.load(ckpt)
    print(f"[mm_{args.slot}] loaded {ckpt}")

    platform = common.connect_platform()
    arena = platform.reconnect_arena(state["arena_id"])
    conn = arena.connect_trader(args.slot, session_tag=7000 + args.slot)
    asset_id = state["asset_id"]
    conn.activate(asset_id)
    market = common.MarketSource(platform)
    print(f"[mm_{args.slot}] md status: {market.status()}")

    logger = common.CycleLogger(args.out, f"mm_{args.slot}", LOG_FIELDS)
    ack_logger = common.CycleLogger(
        args.out, f"mm_{args.slot}_ack",
        ["cycle", "tag", "side", "status", "reject", "price", "remaining"])
    joiner = MMJoiner(agent, conn, market, asset_id, raw_per_price, qty_per_unit,
                      ack_logger, ref_price=float(cfg["ref_price"]),
                      ref_anchor_warmup=bool(cfg.get("ref_anchor_warmup", False)))
    rounds = int(state["rounds"])
    cycle_secs = float(cfg["cycle_secs"])
    rng = random.Random(common.seed_for(cfg["seed"], "mm", args.slot) + 7)

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    t0 = float(state.get("t_start", 0.0)) or time.time() + 5.0
    t0 = max(t0, time.time())
    md0 = map_md(market.get_latest(asset_id, attempts=3), raw_per_price, qty_per_unit)
    agent.on_episode_start(md0)
    cycle = 0
    try:
        while running and cycle < rounds:
            t_next = t0 + (cycle + 1) * cycle_secs + rng.uniform(0.0, float(cfg["jitter_secs"]))
            delay = t_next - time.time()
            if delay > 0:
                time.sleep(delay)
            logger.row(**joiner.run_cycle(cycle))
            cycle += 1
    finally:
        try:
            conn.cancel_all(asset_id)
            conn.drain_acks()
            time.sleep(0.2)
        except Exception as e:
            print(f"[mm_{args.slot}] cleanup warning: {e}")
        logger.close()
        print(f"[mm_{args.slot}] done after {cycle} cycles")


if __name__ == "__main__":
    main()
