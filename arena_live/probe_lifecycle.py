"""Isolate why the arena leaves Trade after the first trade.

Sequence: create arena -> trade -> post bid -> ack -> trade into it ->
lifecycle -> post again -> ack -> lifecycle. Destroys the arena at the end.

Usage: python -u probe_lifecycle.py
"""

from __future__ import annotations

import time

from joiners import common


def dump(tag, obj):
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            obj = dataclasses.asdict(obj)
            print(f"  [{tag}] {obj}")
            return
    except Exception:
        pass
    print(f"  [{tag}] {obj}")


def drain(conn, tag):
    for a in conn.drain_acks():
        dump(f"{tag} ack", a)
    for f in conn.drain_fills():
        dump(f"{tag} fill", f)


def main():
    from fastboxex.price import from_float

    p = common.connect_platform()
    ms = common.MarketSource(p)

    def lifecycle(aid):
        md = ms.get_latest(aid)
        if md is None:
            return "no-md"
        return (f"lifecycle={md.get('lifecycle_state')} bid={md.get('bid')} "
                f"ask={md.get('ask')} src={ms.src}")

    arena = p.create_arena(
        "PRB",
        num_traders=1,
        reference_price=from_float(1000.0),
        tick_size=100,
        lot_size=100,
        generator_cash=100_000_000,
        generator_inventory=1_000_000,
        trader_cash=100_000_000,
        trader_inventory=2000,
        max_order_qty=50000,
        circuit_breaker_bps=2000,
        maker_fee_bps=0,
        taker_fee_bps=0,
        exchange_fee_bps=0,
        lifetime_minutes=30,
    )
    aid = arena.asset_id
    print(f"arena {arena.arena_id} asset {aid}")
    try:
        arena.transition_to("suspended"); time.sleep(0.3)
        arena.transition_to("book_only"); time.sleep(0.3)
        arena.transition_to("trade"); time.sleep(0.5)

        gen = arena.connect_generator(session_tag=1)
        gen.activate(aid)
        tr = arena.connect_trader(0, session_tag=2)
        tr.activate(aid)
        time.sleep(0.5)

        print("\n-- 1: post bid @993 (limit) --")
        tr.submit_limit(aid, 0, 99300, 100, 101)
        time.sleep(1.0)
        drain(tr, "t1")

        print("\n-- lifecycle after post --")
        print("  ", lifecycle(aid))

        print("\n-- 2: generator SELLS 1 unit into the bid --")
        gen.submit_market(aid, 1, 100, 501)
        time.sleep(1.0)
        drain(gen, "g1")
        drain(tr, "t2")

        print("\n-- lifecycle after trade --")
        print("  ", lifecycle(aid))

        print("\n-- 3: post bid again @990 --")
        tr.submit_limit(aid, 0, 99000, 100, 102)
        time.sleep(1.0)
        drain(tr, "t3")

        print("\n-- 4: post ask @1007 (has 20 units inventory) --")
        tr.submit_limit(aid, 1, 100700, 100, 103)
        time.sleep(1.0)
        drain(tr, "t4")

        print("\n-- lifecycle at end --")
        print("  ", lifecycle(aid))

        print("\n-- 5: post after 5s idle --")
        time.sleep(5.0)
        tr.submit_limit(aid, 0, 98900, 100, 104)
        time.sleep(1.0)
        drain(tr, "t5")
        print("  ", lifecycle(aid))
    finally:
        p.close_arena(arena.arena_id)
        print(f"\narena {arena.arena_id} destroyed")


if __name__ == "__main__":
    main()
