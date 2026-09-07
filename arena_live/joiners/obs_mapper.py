"""Live market-data dict → sandbox agent md dict.

The sandbox agents read an md dict with keys bid/ask/bids/asks and cumulative
taker-volume counters (see PPOHjbMarketMaker._update_lambda_est and
KyleInformedTrader.build_observation). fastboxex Market.get_latest() provides
the same information:

    bid, bid_qty, ask, ask_qty, bids [(price, qty)], asks, volume_buy,
    volume_sell  (raw price ticks; volumes in lots)

Prices are converted from raw exchange ticks to the agents' display scale
(raw / raw_per_tick); volumes pass through (lot_size = 1).
"""

from __future__ import annotations


def map_md(raw: dict | None, raw_per_tick: int, qty_per_unit: int = 1) -> dict | None:
    if not raw:
        return None
    r = float(raw_per_tick)
    q = float(qty_per_unit)
    bid = raw.get("bid") or 0
    ask = raw.get("ask") or 0
    bids = [(float(p) / r, float(v) / q) for p, v in (raw.get("bids") or [])]
    asks = [(float(p) / r, float(v) / q) for p, v in (raw.get("asks") or [])]
    return {
        "bid": float(bid) / r if bid else 0.0,
        "ask": float(ask) / r if ask else 0.0,
        "bid_qty": float(raw.get("bid_qty") or 0.0) / q,
        "ask_qty": float(raw.get("ask_qty") or 0.0) / q,
        "bids": bids,
        "asks": asks,
        "taker_volume_buy": int((raw.get("volume_buy") or 0) / q),
        "taker_volume_sell": int((raw.get("volume_sell") or 0) / q),
    }
