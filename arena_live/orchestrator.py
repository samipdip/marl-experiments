"""Arena orchestrator: create arena, spawn joiners, run noise bot, clean up.

Lifecycle: login -> create_arena (tick 100 raw / display unit, ref from_float(1000),
lot 1, fees 0) -> write session state -> suspended -> book_only -> trade ->
spawn MM joiners + BeliefTrader joiners (fixed slots; bias_mode=paired ->
orchestrator draws net-zero +/- bias pairs and passes --bias) -> run the Poisson
noise bot on the generator connection in this process; churn scenarios
(biased/neutral) respawn belief traders with fresh biases -> destroy arena
(always, in finally).

Usage:
    python orchestrator.py --scenario calm            # full session
    python orchestrator.py --scenario calm --rung 1   # 1 MM + noise smoke
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from joiners import common
from joiners.noise_generator import run_noise_loop

PKG_ROOT = pathlib.Path(__file__).resolve().parent


def draw_life(rng, mean: float, lo: int, hi: int) -> int:
    """Trader lifetime in cycles: exponential around `mean`, clamped to [lo, hi]."""
    life = int(rng.expovariate(1.0 / max(mean, 1.0)))
    return max(lo, min(hi, life))


def draw_paired_biases(rng, n: int, bias_max: int) -> list[int]:
    """Net-zero bias set for n trader slots: consecutive slots form +/- pairs.

    m ~ U{1..bias_max} per pair, sign order shuffled per pair -> sum == 0 for
    even n (odd n leaves one 0 = trader at v = F). Cross-sectional mean(v_i)
    equals F exactly, so scenario effect is not confounded by draw luck.
    """
    biases: list[int] = []
    for _ in range(n // 2):
        m = rng.randint(1, max(1, int(bias_max)))
        pair = [m, -m]
        rng.shuffle(pair)
        biases.extend(pair)
    if n % 2:
        biases.append(0)
    return biases


class PairedBiasQueue:
    """Churn-mode net-zero bias stream: pops +/- pairs in random sign order."""

    def __init__(self, rng, bias_max: int):
        self.rng = rng
        self.bias_max = int(bias_max)
        self.q: list[int] = []

    def draw(self) -> int:
        if not self.q:
            m = self.rng.randint(1, max(1, self.bias_max))
            pair = [m, -m]
            self.rng.shuffle(pair)
            self.q.extend(pair)
        return self.q.pop(0)


def main() -> None:
    ap = argparse.ArgumentParser(description="g99 live-arena orchestrator")
    ap.add_argument("--scenario", required=True,
                    help="scenario name from config/scenarios.yaml")
    ap.add_argument("--config", default="config/scenarios.yaml")
    ap.add_argument("--rung", type=int, default=3, help="1=MM smoke, 3=full session")
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--hold-secs", type=float, default=0.0,
                    help="pause after arena creation (BookOnly) so the TUI can be attached")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg seed (validation runs: fresh bias/arrival/"
                         "noise draws per session instead of identical replicates)")
    args = ap.parse_args()

    cfg = common.load_scenario(args.scenario, args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    rounds = args.rounds or int(cfg["rounds"])
    rung1 = args.rung == 1
    n_mms = 1 if rung1 else int(cfg["n_mms"])

    churn_cfg = dict(cfg.get("churn") or {})
    churn_on = bool(churn_cfg.get("enabled")) and not rung1
    churn_slots = int(churn_cfg.get("slots", cfg.get("n_traders", 4)))
    churn_parallel = int(churn_cfg.get("n_parallel", churn_slots))
    churn_start_after = int(churn_cfg.get("start_after", 30))
    churn_life_mean = float(churn_cfg.get("life_mean", 60))
    churn_life_min = int(churn_cfg.get("life_min", 20))
    churn_life_max = int(churn_cfg.get("life_max", 150))
    churn_gap = int(churn_cfg.get("respawn_gap", 8))
    n_traders = 0 if churn_on else (0 if rung1 else int(cfg["n_traders"]))

    bias_mode = str(cfg.get("bias_mode", "random"))
    # Separate stream from the churn orchestrator rng (slot 0) so legacy
    # random-mode churn draws stay reproducible.
    import random as _rand
    bias_rng = _rand.Random(common.seed_for(cfg["seed"], "orch", 1))
    fixed_biases = (draw_paired_biases(bias_rng, n_traders, int(cfg["bias_max"]))
                    if bias_mode == "paired" and n_traders > 0 else None)

    session = args.session or f"{args.scenario}_rung{args.rung}_{int(time.time())}"
    out_root = PKG_ROOT / "runs" / session
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / ".arena_state.json"

    from fastboxex import Platform, load_config
    from fastboxex.price import from_float, pct_to_bps

    common.load_env()
    cfg_fb = load_config()
    username = os.environ.get("FASTBOXEX_USER", "") or cfg_fb.username or ""
    password = os.environ.get("FASTBOXEX_PASS", "")
    if not password:
        import getpass
        password = getpass.getpass(f"Password for {username}: ")
    print(f"[orch] logging in as {username} ...")
    p = Platform.login(cfg_fb, username, password)

    ac = cfg["arena"]
    arena = p.create_arena(
        "G99",
        num_traders=n_mms + n_traders + (churn_slots if churn_on else 0),
        reference_price=from_float(float(cfg["ref_price"])),
        tick_size=int(cfg["tick_size"]),
        lot_size=int(cfg["lot_size"]),
        generator_cash=int(ac["generator_cash"]),
        generator_inventory=int(ac["generator_inventory"]),
        trader_cash=int(ac["trader_cash"]),
        trader_inventory=int(ac["trader_inventory"]),
        max_order_qty=int(ac["max_order_qty"]),
        circuit_breaker_bps=int(ac["circuit_breaker_bps"]),
        maker_fee_bps=int(ac["maker_fee_bps"]),
        taker_fee_bps=int(ac["taker_fee_bps"]),
        exchange_fee_bps=int(ac["exchange_fee_bps"]),
        lifetime_minutes=int(ac["lifetime_minutes"]),
    )
    asset_id = arena.asset_id
    try:
        import dataclasses
        rep = dataclasses.asdict(arena) if dataclasses.is_dataclass(arena) else {
            k: getattr(arena, k) for k in dir(arena)
            if not k.startswith("_") and isinstance(getattr(arena, k), (int, float, str, bool))}
        (out_root / "arena_report.json").write_text(json.dumps(rep, indent=2, default=str))
    except Exception as e:
        print(f"[orch] arena report dump failed: {e}")
    hold = max(float(args.hold_secs), 0.0)
    state = {
        "arena_id": arena.arena_id,
        "asset_id": asset_id,
        "scenario": args.scenario,
        "f": float(cfg["f"]),
        "seed": int(cfg["seed"]),
        "rounds": rounds,
        "rung": args.rung,
        "out_root": str(out_root),
        "t_start": time.time() + hold + 30.0,
    }
    state_path.write_text(json.dumps(state, indent=2))
    meta = {k: v for k, v in state.items() if k != "out_root"}
    (out_root / "session_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[orch] arena {arena.arena_id} asset {asset_id} -> {state_path}")

    procs: list[subprocess.Popen] = []
    arena_handle = arena
    try:
        arena.transition_to("suspended")
        time.sleep(0.3)
        arena.transition_to("book_only")
        time.sleep(0.5)
        if hold > 0:
            print(f"[orch] HOLDING {hold:.0f}s in BookOnly -- attach the TUI now:")
            print(f"[orch]   on QFRCAI:  tmux attach -t tui")
            print(f"[orch]   from local: bash watch_tui_local.sh")
            print(f"[orch]   invite code: {arena.invite_code}")
            try:
                subprocess.run(["tmux", "kill-session", "-t", "tui"],
                               capture_output=True, timeout=10)
                time.sleep(0.5)
                subprocess.Popen(
                    ["bash", str(PKG_ROOT / "watch_tui.sh")],
                    cwd=PKG_ROOT, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True)
                print("[orch] TUI refreshed (fresh dashboard picks up this arena; "
                      "if it shows a password prompt, type it after attaching)")
            except Exception as e:
                print(f"[orch] TUI refresh skipped: {e}")
            time.sleep(hold)
        arena.transition_to("trade")
        time.sleep(0.5)
        print(f"[orch] arena in TRADE")

        gen_conn = arena.connect_generator(session_tag=1)
        gen_conn.activate(asset_id)
        ms = common.MarketSource(p)

        py = [sys.executable, "-u"]
        for i in range(n_mms):
            out = out_root / f"mm_{i}"
            out.mkdir(exist_ok=True)
            logf = open(out / "stdout.log", "w")
            procs.append(subprocess.Popen(
                [*py, "-m", "joiners.mm_joiner", "--scenario", args.scenario,
                 "--slot", str(i), "--config", args.config,
                 "--state", str(state_path), "--out", str(out)],
                cwd=PKG_ROOT, stdout=logf, stderr=subprocess.STDOUT))
        bias_fh = None
        if fixed_biases is not None:
            bias_fh = open(out_root / "biases.csv", "w")
            bias_fh.write("ts,cycle,event,slot,label,bias\n")
            print(f"[orch] bias_mode=paired fixed biases: {fixed_biases} "
                  f"(sum=0, mean v = F={cfg['f']})")
        for i in range(n_traders):
            out = out_root / f"belief_{i}"
            out.mkdir(exist_ok=True)
            logf = open(out / "stdout.log", "w")
            cmd = [*py, "-m", "joiners.belief_joiner", "--scenario", args.scenario,
                   "--slot", str(i), "--config", args.config,
                   "--state", str(state_path), "--out", str(out)]
            if fixed_biases is not None:
                cmd += ["--bias", str(fixed_biases[i])]
                bias_fh.write(f"{time.time()},0,assign,{i},belief_{i},"
                              f"{fixed_biases[i]}\n")
            procs.append(subprocess.Popen(cmd, cwd=PKG_ROOT, stdout=logf,
                                          stderr=subprocess.STDOUT))
        if bias_fh is not None:
            bias_fh.flush()
        print(f"[orch] spawned {len(procs)} fixed joiners"
              + ("; churn manager ON for belief traders" if churn_on else ""))

        # ── Master loop: noise stepping + belief-trader churn ──────────────
        from joiners.noise_generator import NoiseBot
        from joiners.obs_mapper import map_md
        import random as _random

        noise = NoiseBot(common.seed_for(cfg["seed"], "noise", 0),
                         lam=float(cfg.get("noise_lambda", 1.0)),
                         qty_per_unit=int(cfg.get("qty_per_unit", 1)))
        orch_rng = _random.Random(common.seed_for(cfg["seed"], "orch", 0))
        raw_per_price = int(cfg["raw_per_price"])
        qty_per_unit = int(cfg.get("qty_per_unit", 1))
        t0 = float(state["t_start"])

        churn_fh = open(out_root / "churn_log.csv", "w")
        churn_fh.write("ts,cycle,event,slot,spawn_id,bias,life_cycles,exit_code\n")

        # Churn bias source: paired queue (net-zero +/-) or legacy random draws.
        paired_bias_q = (PairedBiasQueue(bias_rng, int(cfg["bias_max"]))
                         if bias_mode == "paired" else None)

        def record_bias(cycle_val, slot_val, label_val, bias_val):
            nonlocal bias_fh
            if bias_fh is None:
                bias_fh = open(out_root / "biases.csv", "w")
                bias_fh.write("ts,cycle,event,slot,label,bias\n")
            bias_fh.write(f"{time.time()},{cycle_val},spawn,{slot_val},"
                          f"{label_val},{bias_val}\n")
            bias_fh.flush()

        active: dict[int, dict] = {}   # slot -> {proc, logf, spawn_id, since}
        free_at = {s: 0 for s in range(churn_slots)} if churn_on else {}
        spawn_counter = 0

        noise_fh = open(out_root / "noise_log.csv", "w")
        noise_fh.write("cycle,ts,n_buy,n_sell,fills_seen\n")
        # Market snapshot per cycle (display units, same lob_json shape as the
        # sandbox checkpoints/*_lob_data.csv): top-5 levels per side + the
        # cumulative taker-volume counters. Phase note: taken at the orchestrator
        # cycle boundary, before this cycle's noise orders.
        md_fh = open(out_root / "md_log.csv", "w")
        md_fh.write("cycle,ts,md_ok,bid,ask,mid,lob_json,taker_buy,taker_sell\n")
        for cycle in range(rounds):
            t_next = t0 + (cycle + 1) * float(cfg["cycle_secs"]) \
                + orch_rng.uniform(0.0, float(cfg["jitter_secs"]))
            delay = t_next - time.time()
            if delay > 0:
                time.sleep(delay)

            md_raw = ms.get_latest(asset_id)
            md = map_md(md_raw, raw_per_price, qty_per_unit) if md_raw else None
            if md is not None:
                m_val = ((md["bid"] + md["ask"]) / 2.0
                         if md["bid"] > 0 and md["ask"] > 0 else "")
                lob_json = json.dumps({
                    "bids": [[p, q] for p, q in md["bids"][:5]],
                    "asks": [[p, q] for p, q in md["asks"][:5]],
                })
                lob_csv = '"' + lob_json.replace('"', '""') + '"'
                md_fh.write(f"{cycle},{time.time()},1,{md['bid']},{md['ask']},"
                            f"{m_val},{lob_csv},{md['taker_volume_buy']},"
                            f"{md['taker_volume_sell']}\n")
            else:
                md_fh.write(f"{cycle},{time.time()},0,,,,,,,\n")
            md_fh.flush()
            n_buy, n_sell = noise.cycle(gen_conn, asset_id, md,
                                        float(cfg.get("market_aggression", 5.0)),
                                        float(cfg["ref_price"]), raw_per_price)
            noise.drain(gen_conn)
            noise_fh.write(f"{cycle},{time.time()},{n_buy},{n_sell},{noise.fills_seen}\n")
            noise_fh.flush()

            if churn_on:
                # reap exited traders
                for slot in list(active):
                    rec = active[slot]
                    rc = rec["proc"].poll()
                    if rc is not None:
                        churn_fh.write(f"{time.time()},{cycle},exit,{slot},"
                                       f"{rec['spawn_id']},,{rc}\n")
                        churn_fh.flush()
                        rec["logf"].close()
                        free_at[slot] = cycle + churn_gap
                        del active[slot]
                # spawn replacements (population maintenance)
                if cycle >= churn_start_after and (rounds - cycle) >= churn_life_min:
                    def _spawn(slot, bias, life):
                        nonlocal spawn_counter
                        spawn_counter += 1
                        record_bias(cycle, slot, f"belief_s{spawn_counter}", bias)
                        out = out_root / f"belief_s{spawn_counter}"
                        out.mkdir(exist_ok=True)
                        logf = open(out / "stdout.log", "w")
                        proc = subprocess.Popen(
                            [*py, "-m", "joiners.belief_joiner",
                             "--scenario", args.scenario, "--slot", str(slot),
                             "--config", args.config, "--state", str(state_path),
                             "--out", str(out), "--rounds", str(life),
                             "--spawn-id", str(spawn_counter), "--bias", str(bias),
                             "--cycle-offset", str(cycle)],
                            cwd=PKG_ROOT, stdout=logf, stderr=subprocess.STDOUT)
                        active[slot] = {"proc": proc, "logf": logf,
                                        "spawn_id": spawn_counter, "since": cycle}
                        procs.append(proc)
                        churn_fh.write(f"{time.time()},{cycle},spawn,{slot},"
                                       f"{spawn_counter},{bias},{life},\n")
                        churn_fh.flush()
                        free_at[slot] = 10**9  # occupied until exit reaps it

                    couple = bool(churn_cfg.get("couple_pairs")) \
                        and paired_bias_q is not None
                    while len(active) < churn_parallel \
                            and any(f <= cycle for f in free_at.values()):
                        free_slots = sorted(s for s, f in free_at.items()
                                            if f <= cycle)
                        if couple:
                            # +/- pair spawns together, identical lifetime ->
                            # exits together; alive-set stays net-zero.
                            if len(free_slots) < 2 or len(active) > churn_parallel - 2:
                                break
                            life = min(draw_life(orch_rng, churn_life_mean,
                                                 churn_life_min, churn_life_max),
                                       rounds - cycle)
                            b1, b2 = paired_bias_q.draw(), paired_bias_q.draw()
                            _spawn(free_slots[0], b1, life)
                            _spawn(free_slots[1], b2, life)
                        else:
                            life = min(draw_life(orch_rng, churn_life_mean,
                                                 churn_life_min, churn_life_max),
                                       rounds - cycle)
                            bias = (paired_bias_q.draw()
                                    if paired_bias_q is not None
                                    else orch_rng.randint(int(cfg["bias_min"]),
                                                          int(cfg["bias_max"])))
                            _spawn(free_slots[0], bias, life)
        noise_fh.close()
        md_fh.close()
        churn_fh.close()
        if bias_fh is not None:
            bias_fh.close()
        n_spawned = spawn_counter

        max_wait = rounds * (float(cfg["cycle_secs"]) + float(cfg["jitter_secs"])) + 120
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if all(pr.poll() is not None for pr in procs):
                break
            time.sleep(5.0)
        for pr in procs:
            if pr.poll() is None:
                print(f"[orch] terminating straggler pid={pr.pid}")
                pr.terminate()
        for pr in procs:
            try:
                pr.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pr.kill()
    finally:
        for pr in procs:
            if pr.poll() is None:
                pr.terminate()
        try:
            p.close_arena(arena_handle.arena_id)
            print(f"[orch] arena {arena_handle.arena_id} destroyed")
        except Exception as e:
            print(f"[orch] cleanup warning: {e}")
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
        print(f"[orch] session {session} complete -> {out_root}")


if __name__ == "__main__":
    main()
