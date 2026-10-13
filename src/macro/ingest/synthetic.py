"""Synthetic data generator (calendar + news + hourly prices) for OFFLINE DEMO / CI only.

It exists so the whole pipeline can be exercised end-to-end without API keys or network.
Numbers produced from this data are NOT real market results and must never be reported as such.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..config import epoch_s
from .calendar_parse import classify_event

ET = "America/New_York"
# title: (impact, base level, surprise sd, unit, decimals)
SPECS = {
    "Unemployment Claims": ("medium", 230, 12, "K", 0),
    "Non-Farm Employment Change": ("high", 150, 60, "K", 0),
    "Unemployment Rate": ("high", 4.2, 0.1, "%", 1),
    "Average Hourly Earnings m/m": ("high", 0.3, 0.1, "%", 1),
    "CPI m/m": ("high", 0.3, 0.1, "%", 1),
    "Core CPI m/m": ("high", 0.3, 0.08, "%", 1),
    "CPI y/y": ("high", 3.0, 0.15, "%", 1),
    "PPI m/m": ("medium", 0.2, 0.2, "%", 1),
    "Retail Sales m/m": ("high", 0.4, 0.5, "%", 1),
    "ISM Manufacturing PMI": ("high", 50.0, 1.5, "", 1),
    "ISM Services PMI": ("high", 52.0, 1.5, "", 1),
    "Core PCE Price Index m/m": ("high", 0.25, 0.08, "%", 2),
    "Advance GDP q/q": ("high", 2.0, 0.6, "%", 1),
    "Prelim UoM Consumer Sentiment": ("medium", 70.0, 3.0, "", 1),
}
FOMC = ["2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
        "2026-09-16", "2026-10-28", "2026-12-09"]
BETA = {  # bps per +1 sd "hawkish/strong" shock, total over the reaction
    "hawk": {"ES=F": -10, "NQ=F": -14, "ZN=F": -7, "GC=F": -8, "EURUSD=X": -6, "CL=F": -2},
    "growth": {"ES=F": 5, "NQ=F": 6, "ZN=F": -5, "GC=F": -3, "EURUSD=X": -4, "CL=F": 5},
}
HVOL = {"ES=F": 10, "NQ=F": 14, "ZN=F": 4, "GC=F": 9, "EURUSD=X": 4, "CL=F": 18}
P0 = {"ES=F": 5400, "NQ=F": 19000, "ZN=F": 110, "GC=F": 2500, "EURUSD=X": 1.10, "CL=F": 72}
POS_T = ["Stocks rally as growth data beats expectations", "Economy shows resilient expansion, soft landing hopes build",
         "Cooling inflation strengthens case for a rate cut", "Hiring rebound lifts optimism on the US labor market",
         "Manufacturing improves, markets upbeat on growth outlook"]
NEG_T = ["Recession fears mount as factory activity contracts", "Layoffs rise, labor market weaker than expected",
         "Stocks sell-off as growth slowdown worries deepen", "Weak consumer spending raises downturn concerns",
         "Tariff worries and slump in orders hit economic outlook"]
HAWK_T = ["Fed officials signal higher for longer as sticky inflation persists", "Hot inflation reading fuels hawkish rate hike bets"]
DOVE_T = ["Dovish Fed tone as inflation cools, rate cuts back on the table", "Disinflation trend supports easing, Powell stays patient"]


def _wd(d: pd.Timestamp) -> pd.Timestamp:
    while d.dayofweek >= 5:
        d += pd.Timedelta(days=1)
    return d


def _schedule(start: pd.Timestamp, end: pd.Timestamp):
    occ = []
    for p in pd.period_range(start, end, freq="M"):
        first = pd.Timestamp(p.year, p.month, 1)
        ff = first + pd.Timedelta(days=(4 - first.dayofweek) % 7)
        for t in ["Non-Farm Employment Change", "Unemployment Rate", "Average Hourly Earnings m/m"]:
            occ.append((ff, "08:30", t))
        for t in ["CPI m/m", "Core CPI m/m", "CPI y/y"]:
            occ.append((_wd(first + pd.Timedelta(days=11)), "08:30", t))
        occ.append((_wd(first + pd.Timedelta(days=13)), "08:30", "PPI m/m"))
        occ.append((_wd(first + pd.Timedelta(days=14)), "08:30", "Retail Sales m/m"))
        occ.append((_wd(first), "10:00", "ISM Manufacturing PMI"))
        occ.append((_wd(first + pd.Timedelta(days=2)), "10:00", "ISM Services PMI"))
        occ.append((_wd(first + pd.Timedelta(days=27)), "08:30", "Core PCE Price Index m/m"))
        occ.append((ff + pd.Timedelta(days=7), "10:00", "Prelim UoM Consumer Sentiment"))
        if p.month in (1, 4, 7, 10):
            occ.append((_wd(first + pd.Timedelta(days=27)), "08:30", "Advance GDP q/q"))
    for d in pd.date_range(start, end, freq="D"):
        if d.dayofweek == 3:
            occ.append((d, "08:30", "Unemployment Claims"))
    return [o for o in occ if start <= o[0] <= end]


def make_synthetic(cfg: dict, now: pd.Timestamp):
    rng = np.random.default_rng(cfg["project"]["seed"])
    sig = float(cfg.get("synthetic", {}).get("signal_scale", 1.0))   # test hook: strengthen the injected effect
    lb, wu = cfg["project"]["lookback_days"], cfg["project"]["warmup_days"]
    start = (now.tz_convert(None).normalize() - pd.Timedelta(days=lb + wu))
    end = now.tz_convert(None).normalize() + pd.Timedelta(days=21)
    occ = sorted(_schedule(start, end), key=lambda o: (o[0], o[1]))

    items, level, prev, shocks = [], {}, {}, {}
    nid = 1000
    for d, hhmm, title in occ:
        imp, base, sd, unit, dec = SPECS[title]
        ts = pd.Timestamp(f"{d:%Y-%m-%d} {hhmm}", tz=ET).tz_convert("UTC")
        lvl = level.get(title, base)
        fc = round(lvl + rng.normal(0, sd * 0.3), dec)
        sur = rng.normal(0, sd)
        act = round(fc + sur, dec)
        level[title] = 0.5 * lvl + 0.5 * act
        fmt = (lambda v: f"{v:.{dec}f}{'%' if unit == '%' else ('K' if unit == 'K' else '')}")
        released = ts <= now
        cat, pol, _ = classify_event(title)
        if released:
            shocks.setdefault(ts, []).append((cat, pol * (act - fc) / sd))
        items.append({"id": nid, "title": title, "currency": "USD", "impact": imp, "datetimeISO": ts.isoformat(),
                      "actual": fmt(act) if released else None, "forecast": fmt(fc),
                      "previous": fmt(prev.get(title, lvl)), "revision": ""})
        prev[title] = act
        nid += 1
    for d in FOMC:
        ts = pd.Timestamp(f"{d} 14:00", tz=ET).tz_convert("UTC")
        if not (start <= ts.tz_convert(None) <= end + pd.Timedelta(days=1)):
            continue
        released = ts <= now
        sur = rng.choice([-0.25, 0.0, 0.25], p=[0.1, 0.8, 0.1])
        if released and sur != 0:
            shocks.setdefault(ts, []).append(("policy", float(np.sign(sur)) * 2.0))
        items.append({"id": nid, "title": "Federal Funds Rate", "currency": "USD", "impact": "high",
                      "datetimeISO": ts.isoformat(), "actual": f"{4.25 + sur:.2f}%" if released else None,
                      "forecast": "4.25%", "previous": "4.25%", "revision": ""}); nid += 1
        items.append({"id": nid, "title": "FOMC Statement", "currency": "USD", "impact": "high",
                      "datetimeISO": ts.isoformat(), "actual": None, "forecast": "", "previous": "", "revision": ""}); nid += 1
        items.append({"id": nid, "title": "FOMC Press Conference", "currency": "USD", "impact": "high",
                      "datetimeISO": (ts + pd.Timedelta(minutes=30)).isoformat(), "actual": None,
                      "forecast": "", "previous": "", "revision": ""}); nid += 1

    # ---- prices -----------------------------------------------------------------------------
    pstart = now.floor("D") - pd.Timedelta(days=lb + 45)
    hrs = pd.date_range(pstart, now.floor("h"), freq="h")
    hrs = hrs[hrs.dayofweek < 5]
    n = len(hrs)
    ts_arr = epoch_s(hrs)
    # daily latent macro mood (also drives news)
    days = pd.date_range(pstart.floor("D"), now.floor("D"), freq="D")
    mood = np.zeros(len(days))
    day_ix = {d.normalize(): i for i, d in enumerate(days)}
    hawk_mood = np.zeros(len(days))
    for i in range(1, len(days)):
        mood[i] = 0.9 * mood[i - 1] + rng.normal(0, 0.12)
        hawk_mood[i] = 0.9 * hawk_mood[i - 1] + rng.normal(0, 0.12)
    for ts, lst in shocks.items():
        di = day_ix.get(ts.floor("D").tz_convert("UTC"), None)
        if di is None:
            continue
        for cat, z in lst:
            if cat in ("labor", "growth"):
                mood[di:] += 0.10 * z * 0.9 ** np.arange(len(days) - di)
            else:
                mood[di:] -= 0.08 * z * 0.9 ** np.arange(len(days) - di)
                hawk_mood[di:] += 0.15 * z * 0.9 ** np.arange(len(days) - di)
    mood = np.clip(mood, -1.5, 1.5)
    price_rows = []
    for sym in cfg["markets"]["assets"]:
        r = rng.normal(0, HVOL[sym], n)
        for ts, lst in shocks.items():
            i0 = int(np.searchsorted(ts_arr, int(ts.timestamp()), side="right")) - 1  # bar containing the release
            if i0 < 0 or i0 + 5 >= n:
                continue
            for cat, z in lst:
                beta = sig * BETA["hawk" if cat in ("inflation", "policy") else "growth"][sym] * z
                r[i0] += 0.4 * beta
                r[i0 + 1: i0 + 5] += 0.6 * beta / 4  # tradable continuation
        close = P0[sym] * np.exp(np.cumsum(r) / 1e4)
        opn = np.r_[close[0], close[:-1]]
        hi = np.maximum(opn, close) * (1 + np.abs(rng.normal(0, 2e-4, n)))
        lo = np.minimum(opn, close) * (1 - np.abs(rng.normal(0, 2e-4, n)))
        price_rows.append(pd.DataFrame({"symbol": sym, "ts_utc": ts_arr, "open": opn, "high": hi, "low": lo,
                                        "close": close, "volume": 1000.0}))
    prices = pd.concat(price_rows, ignore_index=True)

    # ---- news -------------------------------------------------------------------------------
    news = []
    for i, d in enumerate(days):
        for _ in range(rng.integers(8, 15)):
            pub = d.tz_convert("UTC") if d.tzinfo else d.tz_localize("UTC")
            pub = pub + pd.Timedelta(seconds=int(rng.integers(0, 86400)))
            if pub > now:
                continue
            p_pos = 1 / (1 + np.exp(-2.0 * mood[i]))
            if rng.random() < 0.3:
                p_h = 1 / (1 + np.exp(-2.0 * hawk_mood[i]))
                title = rng.choice(HAWK_T if rng.random() < p_h else DOVE_T)
            else:
                title = rng.choice(POS_T if rng.random() < p_pos else NEG_T)
            news.append({"title": str(title), "content": str(title) + " Analysts weigh inflation, jobs and the Fed outlook.",
                         "url": f"https://example.com/n/{i}/{rng.integers(1e9)}",
                         "published_date": pub.strftime("%a, %d %b %Y %H:%M:%S GMT") if rng.random() > 0.05 else None,
                         "query": "synthetic", "source_api": "synthetic"})
    return items, news, prices
