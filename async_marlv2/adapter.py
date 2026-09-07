"""Adapters: present a fastboxex MarlV2Env as the phase3 (market, conn) pair.

Trading_agents (Agents_phase2) run UNCHANGED on top of this shim, so the same
policy classes (PPOHjbMarketMaker, BeliefTrader, PoissonNoiseTrader,
BeliefCoordinator) trained in the phase3 sandbox can train/eval inside the
stop-the-clock, exchange-native MarlV2Env.

Unit conventions (identical to the live arena joiners, see
arena_live/joiners/obs_mapper.py): exchange-native raw ints, agents see
display units: price_display = price_raw / RAW_PER_PRICE,
qty_units = qty_raw / QTY_PER_UNIT. Market-order touch gating mirrors
arena_live/joiners/common.py::submit_orders (skip when the opposite side is
empty), since the live NoiseBot/belief joiners behave that way.
"""

from __future__ import annotations

RAW_PER_PRICE = 10_000      # 1 display unit = 10,000 raw ticks (tick_size=100)
QTY_PER_UNIT = 100          # 1 agent unit = 100 raw shares (lot_size=100)
DEFAULT_ASSET_ID = 1


def md_to_display(md: dict | None) -> dict | None:
    """Exchange-native snapshot dict -> phase3 md dict (display units).

    Key/shape contract identical to arena_live/joiners/obs_mapper.map_md.
    """
    if not md:
        return None
    r, q = float(RAW_PER_PRICE), float(QTY_PER_UNIT)
    return {
        "bid": md["bid"] / r if md["bid"] else 0.0,
        "ask": md["ask"] / r if md["ask"] else 0.0,
        "bid_qty": md["bid_qty"] / q,
        "ask_qty": md["ask_qty"] / q,
        "bids": [(float(p) / r, float(v) / q) for p, v in md["bids"]],
        "asks": [(float(p) / r, float(v) / q) for p, v in md["asks"]],
        "taker_volume_buy": int(md["volume_buy"] // q),
        "taker_volume_sell": int(md["volume_sell"] // q),
    }


class ConnAdapter:
    """Phase3 SimConnection API over a MarlV2 AgentConnection.

    - limit/market prices and quantities converted display <-> raw
    - market orders are skipped when the opposite touch is empty
      (arena parity: joiners/common.submit_orders does the same live)
    """

    def __init__(self, conn, md_fn):
        self._conn = conn
        self._md_fn = md_fn  # zero-arg callable -> raw snapshot dict or None

    def submit_limit(self, asset_id, side, price, qty, client_tag=0):
        self._conn.submit_limit(
            asset_id=int(asset_id), side=int(side),
            price=int(round(float(price) * RAW_PER_PRICE)),
            qty=int(round(float(qty) * QTY_PER_UNIT)),
            client_tag=int(client_tag),
        )

    def submit_market(self, asset_id, side, qty, client_tag=0):
        need = "ask" if int(side) == 0 else "bid"
        md = self._md_fn()
        if not (md and md.get(need)):
            return  # empty book on that side -> skip (cannot cross)
        self._conn.submit_market(
            asset_id=int(asset_id), side=int(side),
            qty=int(round(float(qty) * QTY_PER_UNIT)),
            client_tag=int(client_tag),
        )

    def cancel_by_tag(self, client_tag, asset_id, side):
        self._conn.cancel_by_tag(
            client_tag=int(client_tag), asset_id=int(asset_id), side=int(side))

    def drain_acks(self):
        return list(self._conn.drain_acks())

    def drain_fills(self):
        out = []
        for f in self._conn.drain_fills():
            f["price"] = f["price"] / RAW_PER_PRICE
            f["volume"] = int(round(f["volume"] / QTY_PER_UNIT))
            out.append(f)
        return out


class MarlV2MarketAdapter:
    """Phase3 `Market` surface over a MarlV2Env (seconds-based view).

    Provides exactly the attributes the phase3 agent contract reads:
    env_dt (s), tick_count, sim_t (s), done, get_md(asset_id),
    add_participant(name) -> ConnAdapter.
    """

    def __init__(self, env, asset_id: int = DEFAULT_ASSET_ID):
        self.env = env
        self._asset_id = asset_id
        self.env_dt = env.env_dt_ns / 1e9
        self._pid_by_name: dict[str, int] = {}
        self._md_conn = None  # set via set_md_conn (any participant conn)

    def set_md_conn(self, conn) -> None:
        """Global-book read handle (any AgentConnection sees the same book)."""
        self._md_conn = conn

    @property
    def tick_count(self) -> int:
        return self.env.tick_count

    @property
    def sim_t(self) -> float:
        return self.tick_count * self.env_dt

    @property
    def done(self) -> bool:
        return self.env.done

    def register(self, name: str, pid: int) -> None:
        self._pid_by_name[name] = pid

    def get_md(self, asset_id=None):
        # global book snapshot; any connection returns the same view
        return md_to_display(self._md_conn.get_md(asset_id or self._asset_id))

    def add_participant(self, name):
        # Agent.reset() calls market.add_participant(name); participants are
        # pre-registered with the env, so this just re-binds a connection.
        from fastboxex import AgentConnection
        pid = self._pid_by_name[name]
        return ConnAdapter(AgentConnection(self.env, pid), self._raw_md)

    def _raw_md(self):
        return self._md_conn.get_md(self._asset_id)
