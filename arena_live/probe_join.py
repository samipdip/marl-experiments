"""Probe: can one account join its own arena multiple times for extra trader slots?

create -> trade -> join x2 -> connect trader 0 of each -> post orders -> destroy.
"""

from __future__ import annotations

import time

from joiners import common


def drain(conn, tag):
    for a in conn.drain_acks():
        print(f"  [{tag} ack] status={a.status} reject={a.reject_reason} price={a.price} tag={a.client_tag}")
    for f in conn.drain_fills():
        print(f"  [{tag} fill] price={f.price} vol={f.volume} side={f.side}")


def main():
    from fastboxex.price import from_float

    p = common.connect_platform()
    arena = p.create_arena(
        "PRB",
        num_traders=2,
        reference_price=from_float(1000.0),
        tick_size=100,
        lot_size=100,
        generator_cash=100_000_000_000,
        generator_inventory=1_000_000,
        trader_cash=10_000_000_000,
        trader_inventory=2000,
        joiner_cash=10_000_000_000,
        joiner_inventory=2000,
        joiner_num_traders=1,
        max_participants=40,
        max_order_qty=50000,
        circuit_breaker_bps=2000,
        maker_fee_bps=0,
        taker_fee_bps=0,
        exchange_fee_bps=0,
        lifetime_minutes=30,
    )
    aid = arena.asset_id
    code = arena.invite_code
    print(f"arena {arena.arena_id} asset {aid} code={code}")
    try:
        arena.transition_to("suspended"); time.sleep(0.3)
        arena.transition_to("book_only"); time.sleep(0.3)
        arena.transition_to("trade"); time.sleep(0.5)

        print("-- join #1 --")
        j1 = p.join_arena(code, num_traders=1)
        c1 = j1.connect_trader(0, session_tag=31)
        c1.activate(aid)
        print(f"  joined: user_id={c1.user_id}")

        print("-- join #2 --")
        j2 = p.join_arena(code, num_traders=1)
        c2 = j2.connect_trader(0, session_tag=32)
        c2.activate(aid)
        print(f"  joined: user_id={c2.user_id}")

        print("-- c1 posts bid 999.0, c2 posts ask 1001.0 --")
        c1.submit_limit(aid, 0, 9_990_000, 100, 11)
        c2.submit_limit(aid, 1, 10_010_000, 100, 21)
        time.sleep(1.5)
        drain(c1, "c1")
        drain(c2, "c2")

        print("-- generator crosses the spread (buy) --")
        gen = arena.connect_generator(session_tag=1)
        gen.activate(aid)
        gen.submit_market(aid, 0, 100, 99)
        time.sleep(1.5)
        drain(gen, "gen")
        drain(c2, "c2b")
        drain(c1, "c1b")
    finally:
        p.close_arena(arena.arena_id)
        print(f"arena {arena.arena_id} destroyed")


if __name__ == "__main__":
    main()
