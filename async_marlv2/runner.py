"""runner.py -- phase3 run_simulation equivalent, driven by MarlV2Env.

Episode layout mirrors price_discovery eval runs (set5_ckpt125000):
  - 240 sim-seconds per episode, 10 ms env tick, agents decide every 1 s at
    per-agent random phases (first wake offsets), so agents act at different
    sub-cycle instants and each acting agent observes the CURRENT book
    (stop-the-clock inside on_tick; the arena's async exposure, locally).

Eval mode: fresh env + fresh agents per episode (phase3 eval parity: the
sandbox ran one process per episode). Training mode: single env, reset every
episode_len_s seconds (phase3 episode_ticks parity), agents persist.
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys

import numpy as np
import torch

# Trading_agents: Agents_phase2 package root (local + QFRCAI layouts)
for cand in (os.path.expanduser("~/Trading_agents"),
             os.path.expanduser("~/Documents/Trading_agents")):
    if os.path.isdir(os.path.join(cand, "Agents_phase2")):
        if cand not in sys.path:
            sys.path.insert(0, cand)
        break
else:
    raise RuntimeError("Trading_agents repo not found")

from Agents_phase2.base import execute_orders  # noqa: E402
from Agents_phase2.RL_Agents.ABIDES_MARL.belief_coordinator import (  # noqa: E402
    BeliefCoordinator,
)
from Agents_phase2.RL_Agents.ABIDES_MARL.belief_trader import BeliefTrader  # noqa: E402
from Agents_phase2.RL_Agents.ABIDES_MARL.poisson_noise_trader import (  # noqa: E402
    PoissonNoiseTrader,
)
from Agents_phase2.RL_Agents.ABIDES_MARL.ppo_hjb_market_maker import (  # noqa: E402
    PPOHjbMarketMaker,
)

from fastboxex import AgentConnection, MarlV2Env  # noqa: E402

from .adapter import ConnAdapter, MarlV2MarketAdapter  # noqa: E402

AGENT_CLASSES = {
    "PPOHjbMarketMaker": PPOHjbMarketMaker,
    "BeliefTrader": BeliefTrader,
    "PoissonNoiseTrader": PoissonNoiseTrader,
    "BeliefCoordinator": BeliefCoordinator,
}

# Arena parity constants (arena_live/config/scenarios.yaml + HANDOFF §3)
TICK_SIZE_RAW = 100          # 0.01 display
LOT_SIZE_RAW = 100           # 1 unit
REF_PRICE_RAW = 10_000_000   # 1000.0 display at RAW_PER_PRICE=10_000 (arena ref)
ENV_DT_NS = 10_000_000       # 10 ms ticks; decision_dt 1 s -> period 100
DECISION_DT_S = 1.0
PHASE_SLOTS = int(DECISION_DT_S * 1e9 / ENV_DT_NS)  # 100
CASH_RAW = 50_000_000_000    # trader/joiner cash (arena parity)
INVENTORY_RAW = 2_000        # trader/joiner endowment (arena parity)
MAX_ORDER_QTY = 50_000
CIRCUIT_BREAKER_BPS = 2_000

NON_TRADING = {"BeliefCoordinator"}


class _NullConn:
    """BeliefCoordinator trades nothing; attach() just needs the object."""

    def drain_acks(self):
        return []

    def drain_fills(self):
        return []


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_param(val):
    if isinstance(val, list) and len(val) == 2 and all(
            isinstance(v, (int, float)) for v in val):
        if all(isinstance(v, int) for v in val):
            return random.randint(val[0], val[1])
        return random.uniform(val[0], val[1])
    return val


def _build_agent(spec: dict, config: dict, results_dir: str):
    cls = AGENT_CLASSES[spec["type"]]
    params = {k: _resolve_param(v) for k, v in spec.get("params", {}).items()}
    params.pop("decision_dt", None)
    train_block = config.get("train", {}) or {}
    params.setdefault("total_timesteps", train_block.get("total_timesteps", 0))
    params.setdefault("eval_freq", train_block.get("eval_freq", 0))
    params.setdefault("eval_episodes", train_block.get("eval_episodes", 30))
    params.setdefault("checkpoint_freq", train_block.get("checkpoint_freq", 0))
    params["mode"] = config.get("mode", "eval")
    params["results_dir"] = results_dir
    params.setdefault("name", spec.get("id", ""))
    agent = cls(**params)
    ckpt = spec.get("checkpoint")
    if ckpt and hasattr(agent, "load"):
        agent.load(ckpt)
    if config.get("mode") == "eval" and hasattr(agent, "set_eval_mode"):
        agent.set_eval_mode(True)
    return agent


class _Session:
    """One live env + agents + callback wiring."""

    def __init__(self, config: dict, seed: int, results_dir: str):
        _seed_all(seed)
        self.rng = np.random.default_rng(seed)
        episode_s = float(config.get("environment", {}).get("sim_duration", 240.0))
        self.episode_ns = int(episode_s * 1e9)
        self.env = MarlV2Env(
            tick_size=TICK_SIZE_RAW, lot_size=LOT_SIZE_RAW,
            reference_price=REF_PRICE_RAW, env_dt_ns=ENV_DT_NS,
            sim_duration_ns=self.episode_ns, mode="parity",
            max_order_qty=MAX_ORDER_QTY,
            circuit_breaker_bps=CIRCUIT_BREAKER_BPS,
        )
        for state in ("Suspended", "Auction", "Trade"):
            self.env.transition_to(state)
        self.market = MarlV2MarketAdapter(self.env)
        self.agents = []
        phases = self.rng.integers(0, PHASE_SLOTS, size=len(config["agents"]))
        for spec, phase in zip(config["agents"], phases):
            agent = _build_agent(spec, config, results_dir)
            pid = self.env.add_participant(
                spec["id"], cash=CASH_RAW, inventory=INVENTORY_RAW,
                decision_dt_ns=ENV_DT_NS, is_collector=False)
            self.market.register(spec["id"], pid)
            if spec["type"] in NON_TRADING:
                conn = _NullConn()
            else:
                conn = ConnAdapter(AgentConnection(self.env, pid),
                                   self.market._raw_md)
                if self.market._md_conn is None:
                    self.market.set_md_conn(AgentConnection(self.env, pid))
            agent.attach(self.market, conn, DECISION_DT_S, phase=int(phase))
            self.env.set_on_tick(pid, self._make_cb(agent))
            self.agents.append(agent)

    def _make_cb(self, agent):
        def cb(_t_ns):
            agent.on_tick(self.market.sim_t)
        return cb

    def run_episode(self):
        while not self.env.done:
            self.env.tick()
        # Agents dump lob/action history on market.done, but that block sits
        # behind each agent's scheduler gate; with per-agent phases the
        # terminal tick is not a firing tick for everyone. Flush explicitly.
        for a in self.agents:
            if getattr(a, "_action_history_saved", True) is False:
                a._save_action_history()
                a._dump_lob_history()
                a._action_history_saved = True

    def run_train(self, episode_ticks: int, total_ticks: int):
        """Continuous training: reset env+agents every episode_ticks ticks
        (phase3 Simulation.run episode_ticks parity).

        phase3 did NOT reset agents at episode boundaries: the PPO buffer
        spans boundaries and fills at 240 stores (an episode stores 239 --
        transitions store on the firing AFTER the action), so update()
        fires mid-episode. Resetting agents here would wipe the buffer at
        239/240 and training would silently never update. RolloutBuffer
        .clear() only zeroes pos (contents persist), so we preserve pos,
        the store counter, and the pending action's z/logprob/value across
        the per-episode agent reset.
        """
        tick = 0
        while tick < total_ticks:
            self.env.tick()
            tick += 1
            if tick % episode_ticks == 0 and tick < total_ticks:
                self.env.reset(seed=0)
                for a in self.agents:
                    buf = getattr(a, "buffer", None)
                    pos = getattr(buf, "pos", None)
                    sib = getattr(a, "_steps_in_buffer", 0)
                    pend = tuple(getattr(a, k, None) for k in (
                        "_last_z", "_last_logprob", "_last_value"))
                    a.reset()
                    if buf is not None and pos is not None:
                        buf.pos = pos  # clear() only zeroes pos; contents persist
                    if hasattr(a, "_steps_in_buffer"):
                        a._steps_in_buffer = sib
                    for k, v in zip(("_last_z", "_last_logprob",
                                     "_last_value"), pend):
                        if v is not None and hasattr(a, k):
                            setattr(a, k, v)


def dump_pnl_history(agents: list, results_dir: str) -> None:
    """phase3 runner.dump_pnl_history equivalent."""
    rows = []
    for a in agents:
        name = getattr(a, "name", str(id(a)))
        m = a.get_metrics() or {}
        for row in m.get("pnl_history") or []:
            rows.append({
                "agent": name, "sim_t": row["sim_t"],
                "inventory": row["inventory"], "cash": row["cash"],
                "mid": row["mid"], "bid": row.get("bid"), "ask": row.get("ask"),
                "bid_flush": row.get("bid_flush"), "ask_flush": row.get("ask_flush"),
                "mid_flush": row.get("mid_flush"), "total_pnl": row["total_pnl"],
                "quote_bid": row.get("quote_bid"), "quote_ask": row.get("quote_ask"),
            })
    if not rows:
        return
    path = os.path.join(results_dir, "pnl_history.csv")
    fields = ["agent", "sim_t", "inventory", "cash", "mid", "bid", "ask",
              "bid_flush", "ask_flush", "mid_flush", "total_pnl",
              "quote_bid", "quote_ask"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def dump_order_events(agents: list, results_dir: str) -> None:
    """phase3 runner.dump_order_events equivalent."""
    rows = []
    for a in agents:
        name = getattr(a, "name", str(id(a)))
        m = a.get_metrics() or {}
        for row in m.get("order_events") or []:
            rows.append({"agent": name, "sim_t": row["sim_t"], "tag": row["tag"],
                         "side": row["side"], "event": row["event"],
                         "price": row["price"]})
    if not rows:
        return
    path = os.path.join(results_dir, "order_events.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["agent", "sim_t", "tag", "side",
                                          "event", "price"])
        w.writeheader()
        w.writerows(rows)


def run_eval(config: dict, n_episodes: int, out_root: str, seed0: int = 10000):
    """n_episodes fresh episodes -> ep_<i>/ dirs with phase3 eval layout."""
    for ep in range(n_episodes):
        epdir = os.path.join(out_root, f"ep_{ep}")
        os.makedirs(epdir, exist_ok=True)
        sess = _Session(config, seed=seed0 + ep, results_dir=epdir)
        sess.run_episode()
        dump_pnl_history(sess.agents, epdir)
        dump_order_events(sess.agents, epdir)
        with open(os.path.join(epdir, "run_meta.json"), "w") as f:
            json.dump({"seed": seed0 + ep, "engine": "marlv2"}, f, indent=2)
        print(f"  eval ep {ep} done")


def run_train(config: dict, results_dir: str, seed: int = 42):
    episode_s = float(config.get("environment", {}).get("sim_duration", 240.0))
    episode_ticks = int(episode_s * 1e9 / ENV_DT_NS)
    total_ticks = int(config.get("train", {}).get("total_timesteps", 0)) * \
        int(DECISION_DT_S * 1e9 / ENV_DT_NS)
    os.makedirs(results_dir, exist_ok=True)
    sess = _Session(config, seed=seed, results_dir=results_dir)
    sess.run_train(episode_ticks, total_ticks)
    return results_dir
