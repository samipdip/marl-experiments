"""Poisson noise bot for the arena generator slot.

The background flow of the sandbox market: each cycle, Poisson(lambda) market
orders per side, qty 1. Runs in the orchestrator process on the generator
connection (created on the main thread — the Rust binding is !Send).

Taker orders use the marketable-limit + immediate-cancel emulation (see
common.submit_orders); the touch comes from the shared MarketSource.
"""

from __future__ import annotations

import time


class NoiseBot:
    def __init__(self, seed: int, lam: float = 1.0, qty: int = 1, tag_start: int = 900_000,
                 qty_per_unit: int = 1):
        import numpy as np
        self.rng = np.random.default_rng(seed)
        self.lam = lam
        self.qty = qty * qty_per_unit
        self.tag = tag_start
        self.fills_seen = 0

    def cycle(self, conn, asset_id: int, md: dict | None,
              aggression: float = 5.0, ref_price: float = 1000.0,
              raw_per_price: int = 10000) -> tuple[int, int]:
        n_buy = int(self.rng.poisson(self.lam))
        n_sell = int(self.rng.poisson(self.lam))
        for _ in range(n_buy):
            self.tag += 1
            self._take(conn, asset_id, 0, md, aggression, ref_price, raw_per_price)
        for _ in range(n_sell):
            self.tag += 1
            self._take(conn, asset_id, 1, md, aggression, ref_price, raw_per_price)
        return n_buy, n_sell

    def _take(self, conn, asset_id: int, side: int, md: dict | None,
              aggression: float, ref_price: float, raw_per_price: int) -> None:
        need = "ask" if side == 0 else "bid"
        if not (md and md.get(need)):
            return  # empty book on that side -> skip
        conn.submit_market(asset_id, side, self.qty, self.tag)

    def drain(self, conn) -> int:
        n = 0
        for _f in conn.drain_fills():
            n += 1
        for _a in conn.drain_acks():
            pass
        self.fills_seen += n
        return n


def run_noise_loop(conn, asset_id: int, seed: int, rounds: int,
                   cycle_secs: float, jitter_secs: float, log_path: str,
                   qty_per_unit: int = 1, t0_epoch: float | None = None,
                   market=None, aggression: float = 5.0,
                   ref_price: float = 1000.0, raw_per_price: int = 10000) -> None:
    bot = NoiseBot(seed, qty_per_unit=qty_per_unit)
    rng_seed = (seed * 31 + 17) % (2**31 - 1)
    import random as _random
    import time as _time
    rng = _random.Random(rng_seed)
    t0 = t0_epoch if t0_epoch is not None else _time.time() + 5.0
    t0 = max(t0, _time.time())
    with open(log_path, "w") as fh:
        fh.write("cycle,ts,n_buy,n_sell,fills_seen\n")
        for cycle in range(rounds):
            t_next = t0 + (cycle + 1) * cycle_secs + rng.uniform(0.0, jitter_secs)
            delay = t_next - _time.time()
            if delay > 0:
                _time.sleep(delay)
            md = market.get_latest(asset_id) if market is not None else None
            n_buy, n_sell = bot.cycle(conn, asset_id, md, aggression,
                                      ref_price, raw_per_price)
            bot.drain(conn)
            fh.write(f"{cycle},{_time.time()},{n_buy},{n_sell},{bot.fills_seen}\n")
            fh.flush()
    print(f"[noise] done after {rounds} cycles, fills_seen={bot.fills_seen}")
