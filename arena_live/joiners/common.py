"""Shared utilities for arena_live joiners: config, state, credentials, orders, logging.

fastboxex is imported lazily (inside connect functions) so mock_test can drive
the joiner logic without the SDK installed.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import re
import sys
import time


def ensure_trading_agents_path() -> str:
    for cand in (os.path.expanduser("~/Trading_agents"),
                 os.path.expanduser("~/Documents/Trading_agents")):
        if os.path.isdir(os.path.join(cand, "Agents_phase2")):
            if cand not in sys.path:
                sys.path.insert(0, cand)
            return cand
    raise RuntimeError("Trading_agents repo not found (~/Trading_agents or ~/Documents/Trading_agents)")


def load_yaml(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_scenario(scenario: str, cfg_path: str) -> dict:
    cfg = load_yaml(cfg_path)
    merged = dict(cfg["defaults"])
    scen = dict(cfg["scenarios"][scenario])
    profile = scen.pop("profile", None)
    if profile:
        merged.update(cfg.get("profiles", {})[profile])
    merged.update(scen)
    merged["scenario"] = scenario
    return merged


def final_checkpoint(ckpt_root: str, dir_prefix: str, file_prefix: str,
                     slot: int) -> str:
    """Newest *_final.pt under <root>/<dir_prefix>_<slot>/ (training step unknown)."""
    import glob as _glob
    d = os.path.join(ckpt_root, f"{dir_prefix}_{slot}")
    cands = _glob.glob(os.path.join(d, f"{file_prefix}_{slot}_*_final.pt"))
    if not cands:
        return os.path.join(d, f"{file_prefix}_{slot}_step_180029_final.pt")

    def _step(p):
        m = re.findall(r"step_(\d+)_final", p)
        return int(m[0]) if m else -1
    return max(cands, key=_step)


def read_state(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def load_env(env_path: str = ".env") -> None:
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def connect_platform() -> "tuple":
    from fastboxex import Platform, load_config
    load_env()
    cfg = load_config()
    username = os.environ.get("FASTBOXEX_USER", "") or cfg.username or ""
    password = os.environ.get("FASTBOXEX_PASS", "") or os.environ.get("FASTBOXEX_PASSWORD", "")
    if not username:
        raise RuntimeError("FASTBOXEX_USER not set (.env) and no Config.toml username")
    if not password:
        import getpass
        password = getpass.getpass(f"Password for {username}: ")
    return Platform.login(cfg, username, password)


def fill_to_dict(fill) -> dict:
    if dataclasses.is_dataclass(fill) and not isinstance(fill, type):
        return dataclasses.asdict(fill)
    return dict(fill)


def ack_to_dict(ack) -> dict:
    if dataclasses.is_dataclass(ack) and not isinstance(ack, type):
        return dataclasses.asdict(ack)
    return dict(ack)


# fastboxex RejectReason codes (codec.py RejectReason IntEnum), server-side ints.
REJECT_NAMES = {
    0: "", 1: "WrongLifecycle", 2: "ZeroQty", 3: "ExceedsMaxQty",
    4: "OutsidePriceBand", 5: "InsufficientCash", 6: "InsufficientInventory",
    7: "CostOverflow", 8: "UserFrozen", 9: "NoLiquidity", 10: "WouldCross",
    11: "BookEmpty", 12: "DeltaOverflow",
}


def reject_name(ack_dict: dict) -> str:
    """Normalize reject_reason (int, IntEnum, or SDK name-string) to a name.

    Empty/0/None -> "" (accepted). InsufficientCash is the cash-wall signal.
    """
    rr = ack_dict.get("reject_reason")
    if rr is None or rr == 0 or rr == "":
        return ""
    if isinstance(rr, str):
        return rr
    try:
        return str(getattr(rr, "name", REJECT_NAMES.get(int(rr), str(rr))))
    except (TypeError, ValueError):
        return str(rr)


def to_raw(price: float, raw_per_tick: int) -> int:
    return int(round(price * raw_per_tick))


def to_display(raw_price: float, raw_per_tick: int) -> float:
    return float(raw_price) / raw_per_tick


def submit_orders(conn, orders, asset_id: int, raw_per_tick: int,
                  qty_per_unit: int = 1, md: dict | None = None,
                  aggression: float = 5.0, ref_price: float = 1000.0) -> list[dict]:
    """Translate sandbox Orders to exchange calls.

    Market orders are emulated the platform-blessed way (02_arena_basics.py):
    a marketable limit at touch +/- aggression, immediately followed by
    cancel_by_tag so the unfilled remainder cannot rest (IOC semantics).
    submit_ioc() is NOT used — its delta_ticks semantics rejected most
    orders in practice (98/480 fills, runaway mid).
    """
    sent = []
    for o in orders:
        rec = {"side": int(o.side), "price": float(o.price), "qty": int(o.qty),
               "type": o.order_type, "tag": o.tag}
        qty_raw = int(o.qty) * qty_per_unit
        if o.order_type == "cancel" and o.tag is not None:
            conn.cancel_by_tag(int(o.tag), asset_id, int(o.side))
        elif o.order_type == "limit":
            conn.submit_limit(asset_id, int(o.side), to_raw(o.price, raw_per_tick),
                              qty_raw, int(o.tag))
        elif o.order_type == "market":
            need = "ask" if o.side == 0 else "bid"
            if not (md and md.get(need)):
                continue  # empty book on that side -> skip (cannot cross)
            conn.submit_market(asset_id, int(o.side), qty_raw, int(o.tag))
        else:
            continue
        rec["sent_ts"] = time.time()
        sent.append(rec)
    return sent


class CycleLogger:
    """CSV writer for per-cycle joiner state; flushes every row."""

    def __init__(self, out_dir: str, name: str, fields: list[str]):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, f"{name}_cycle_log.csv")
        self._fh = open(self.path, "w", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=fields, extrasaction="ignore")
        self._w.writeheader()

    def row(self, **kw) -> None:
        self._w.writerow(kw)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def seed_for(seed: int, kind: str, slot: int) -> int:
    kind_id = {"mm": 1, "belief": 2, "noise": 3, "orch": 0}[kind]
    return (int(seed) * 100_003 + kind_id * 1_013 + int(slot) * 101) % (2**31 - 1)


def wait_for_book(arena_market, asset_id: int, timeout_secs: float = 120.0) -> dict | None:
    deadline = time.monotonic() + timeout_secs
    while time.monotonic() < deadline:
        md = arena_market.get_latest(asset_id)
        if md is not None and (md.get("bid") or 0) > 0 and (md.get("ask") or 0) > 0:
            return md
        time.sleep(0.25)
    return None


class MarketSource:
    """Full-book market data with fallback: arena SUB -> production TCP relay.

    arena_market() (ArenaDistributor) is the canonical arena feed; the
    production relay (p.market(), :12000) also carries arena assets. Whichever
    delivers a non-empty book wins; per-cycle status is exposed for logging.
    """

    def __init__(self, platform):
        self.am = platform.arena_market()
        try:
            self.pm = platform.market()
        except Exception:
            self.pm = None
        self.src = "none"

    def get_latest(self, asset_id: int, attempts: int = 1, retry_secs: float = 0.12) -> dict | None:
        self.ok = False
        for _ in range(max(1, attempts)):
            md = self._get_any(asset_id)
            if md is not None:
                return md
            if attempts > 1:
                time.sleep(retry_secs)
        return None

    def _get_any(self, asset_id: int) -> dict | None:
        try:
            md = self.am.get_latest(asset_id)
            if md and ((md.get("bid") or 0) > 0 or (md.get("ask") or 0) > 0):
                self.src, self.ok = "arena", True
                return md
        except Exception:
            pass
        if self.pm is not None:
            try:
                md = self.pm.get_latest(asset_id)
                if md and ((md.get("bid") or 0) > 0 or (md.get("ask") or 0) > 0):
                    self.src, self.ok = "relay", True
                    return md
            except Exception:
                pass
        return None

    def status(self) -> str:
        a = False
        r = False
        try:
            a = bool(self.am.is_connected())
        except Exception:
            pass
        if self.pm is not None:
            try:
                r = bool(self.pm.is_connected())
            except Exception:
                pass
        return f"arena_sub={int(a)} relay_sub={int(r)} src={self.src}"
