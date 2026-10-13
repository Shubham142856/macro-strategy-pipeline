"""Hourly OHLC for the strategy universe (yfinance). Stored in UTC epoch seconds."""
from __future__ import annotations
import logging
import pandas as pd
from ..config import epoch_s

log = logging.getLogger(__name__)


def fetch_prices(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf
    frames = []
    for sym in cfg["markets"]["assets"]:
        h = yf.Ticker(sym).history(start=start.tz_convert(None).date(), end=(end + pd.Timedelta(days=1)).tz_convert(None).date(),
                                   interval=cfg["markets"].get("bar_interval", "1h"), auto_adjust=False)
        if h.empty:
            raise RuntimeError(f"No hourly data for {sym} (yfinance hourly history is limited to the last ~730 days).")
        h = h.rename(columns=str.lower)
        idx = h.index.tz_convert("UTC") if h.index.tz is not None else h.index.tz_localize("UTC")
        f = pd.DataFrame({"symbol": sym, "ts_utc": epoch_s(idx),
                          "open": h["open"].values, "high": h["high"].values, "low": h["low"].values,
                          "close": h["close"].values, "volume": h.get("volume", 0).values})
        frames.append(f.dropna(subset=["open"]))
        log.info("prices %s: %d bars", sym, len(f))
    return pd.concat(frames, ignore_index=True)
