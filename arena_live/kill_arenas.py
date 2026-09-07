"""List / destroy arenas (quota hygiene). Default is dry-run.

Usage:
    python kill_arenas.py                # list arenas
    python kill_arenas.py --yes          # destroy ALL your arenas
    python kill_arenas.py --arena 123    # destroy one
"""

from __future__ import annotations

import argparse

from joiners import common


def main() -> None:
    ap = argparse.ArgumentParser(description="arena hygiene")
    ap.add_argument("--yes", action="store_true", help="destroy all arenas (no dry-run)")
    ap.add_argument("--arena", type=int, default=None, help="destroy this arena id")
    args = ap.parse_args()

    p = common.connect_platform()
    arenas = p.arenas
    if not arenas:
        print("no arenas")
        return
    print(f"{len(arenas)} arena(s):")
    for aid, a in arenas.items():
        print(f"  arena_id={aid}  asset_id={getattr(a, 'asset_id', '?')}")

    if args.arena is not None:
        p.close_arena(args.arena)
        print(f"destroyed arena {args.arena}")
    elif args.yes:
        for aid in list(arenas):
            p.close_arena(aid)
            print(f"destroyed arena {aid}")
    else:
        print("dry-run only; pass --yes or --arena <id>")


if __name__ == "__main__":
    main()
