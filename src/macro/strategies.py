"""Strategy permutations (asset x direction x holding horizon) and their realised net P&L labels.

Timing rule (no look-ahead): a release at time T is traded at the OPEN of the first hourly bar whose start is
>= T + latency. P&L is volatility-scaled so that a 1-sigma move over the holding horizon costs/earns
`risk_per_trade_bps` of capital, net of round-trip costs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import sym_short


def _asset_arrays(prices: pd.DataFrame, sym: str, vol_lb: int):
    p = prices[prices["symbol"] == sym].sort_values("ts_utc")
    ts, op = p["ts_utc"].values, p["open"].values.astype(float)
    lr = np.r_[np.nan, np.diff(np.log(op))] * 1e4
    vol = pd.Series(lr).rolling(vol_lb, min_periods=vol_lb // 2).std().values
    return ts, op, vol


def build_rows(slots: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    st = cfg["strategy"]
    lat = st["entry_latency_minutes"] * 60
    out = []
    for sym, meta in cfg["markets"]["assets"].items():
        ts, op, vol = _asset_arrays(prices, sym, st["vol_lookback_bars"])
        s_ts = slots["slot_ts"].values.astype("int64")
        ei = np.searchsorted(ts, s_ts + lat, side="left")
        ok = (ei < len(ts)) & (ei >= 24)
        ei_c = np.clip(ei, 0, len(ts) - 1)
        ok &= (ts[ei_c] - s_ts) <= st["max_entry_gap_hours"] * 3600
        ok &= np.isfinite(vol[ei_c]) & (vol[ei_c] > 0)
        pre4 = np.log(op[ei_c] / op[np.clip(ei_c - 4, 0, None)]) * 1e4
        pre24 = np.log(op[ei_c] / op[np.clip(ei_c - 24, 0, None)]) * 1e4
        for h in st["horizons_bars"]:
            xi = ei_c + h
            ok_h = ok & (xi < len(ts))
            xi_c = np.clip(xi, 0, len(ts) - 1)
            ok_h &= (ts[xi_c] - ts[ei_c]) <= (h + 60) * 3600          # tolerate a weekend gap, nothing more
            gross = np.log(op[xi_c] / op[ei_c]) * 1e4
            size = np.clip(st["risk_per_trade_bps"] / (vol[ei_c] * np.sqrt(h)), 0, st["max_leverage"])
            for d in st["directions"]:
                pnl = size * (d * gross - meta["cost_bps"])
                df = pd.DataFrame({
                    "slot_ts": s_ts[ok_h], "asset": sym_short(sym), "direction": d, "horizon": h,
                    "entry_ts": ts[ei_c][ok_h], "exit_ts": ts[xi_c][ok_h], "gross_bps": gross[ok_h],
                    "size": size[ok_h], "pnl_bps": pnl[ok_h], "pre_ret_4h": pre4[ok_h], "pre_ret_24h": pre24[ok_h],
                    "pre_vol_bps": vol[ei_c][ok_h]})
                out.append(df)
    rows = pd.concat(out, ignore_index=True)
    rows["win"] = (rows["pnl_bps"] > 0).astype(int)
    return rows


def market_state_now(prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """As-of-now state per asset (last available bar) for scenario scoring of upcoming releases."""
    st = cfg["strategy"]
    recs = []
    for sym in cfg["markets"]["assets"]:
        ts, op, vol = _asset_arrays(prices, sym, st["vol_lookback_bars"])
        i = len(ts) - 1
        recs.append({"asset": sym_short(sym), "pre_ret_4h": np.log(op[i] / op[i - 4]) * 1e4,
                     "pre_ret_24h": np.log(op[i] / op[i - 24]) * 1e4, "pre_vol_bps": vol[i], "asof_ts": int(ts[i])})
    return pd.DataFrame(recs)
