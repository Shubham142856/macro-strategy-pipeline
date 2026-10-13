"""Event enrichment, release 'slots', time-based features and the SQL join of events <-> news."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .ingest.calendar_parse import CATS, KEY_TAGS

POLICY_NO_VALUE = ["fomc", "fomc_minutes", "fed_speech"]


def enrich_events(ev: pd.DataFrame, now_ts: int) -> pd.DataFrame:
    """Causal surprise statistics: std of surprises uses ONLY earlier releases of the same series."""
    ev = ev.sort_values(["ts_utc", "title"]).copy()
    ev["surprise"] = ev["actual"] - ev["forecast"]
    ev["sur_std"] = ev.groupby("title")["surprise"].transform(lambda s: s.shift().expanding(min_periods=4).std())
    ev["surprise_z"] = (ev["surprise"] / ev["sur_std"].replace(0, np.nan)).clip(-4, 4)
    ev["surprise_score"] = ev["surprise_z"].fillna(np.sign(ev["surprise"]))
    ev["signed_score"] = ev["surprise_score"] * ev["polarity"]       # + = stronger economy / more hawkish
    ev["released"] = ((ev["ts_utc"] <= now_ts) & (ev["actual"].notna() | ev["tag"].isin(POLICY_NO_VALUE))).astype(int)
    return ev


def build_slots(ev: pd.DataFrame, display_tz: str) -> pd.DataFrame:
    """One decision per release timestamp (CPI m/m + Core CPI + CPI y/y are ONE trade slot, not three)."""
    rows = []
    for ts, g in ev.groupby("ts_utc"):
        key = g[g["tag"].isin(KEY_TAGS)]
        pool = key if len(key) else g
        head = pool.assign(_r=(pool["impact"] == "high").astype(int)).sort_values("_r").iloc[-1]["title"]
        r = {"slot_ts": int(ts), "headline": head, "n_events": len(g), "n_high": int((g["impact"] == "high").sum()),
             "released": int(g["released"].max())}
        for c in CATS:
            x = g.loc[g["category"] == c, "signed_score"].dropna()
            r[f"s_{c}"] = float(x.mean()) if len(x) else 0.0
        for t in KEY_TAGS:
            r[f"is_{t}"] = int((g["tag"] == t).any())
        rows.append(r)
    s = pd.DataFrame(rows).sort_values("slot_ts").reset_index(drop=True)
    t = pd.to_datetime(s["slot_ts"], unit="s", utc=True).dt.tz_convert(display_tz)
    hr = t.dt.hour + t.dt.minute / 60
    s["hour_et"] = hr
    s["hour_sin"], s["hour_cos"] = np.sin(2 * np.pi * hr / 24), np.cos(2 * np.pi * hr / 24)
    s["dow"] = t.dt.dayofweek
    s["dow_sin"], s["dow_cos"] = np.sin(2 * np.pi * s["dow"] / 7), np.cos(2 * np.pi * s["dow"] / 7)
    s["dom_frac"] = t.dt.day / t.dt.days_in_month
    ts_arr = s["slot_ts"].values
    prev = np.r_[np.nan, np.diff(ts_arr)] / 3600
    nxt = np.r_[np.diff(ts_arr), np.nan] / 3600
    s["hrs_since_prev"] = np.clip(np.nan_to_num(prev, nan=168), 0, 168)   # schedule is public => 'next' is not leakage
    s["hrs_to_next"] = np.clip(np.nan_to_num(nxt, nan=168), 0, 168)
    fomc = np.sort(s.loc[s["is_fomc"] == 1, "slot_ts"].values)
    if len(fomc):
        i = np.searchsorted(fomc, ts_arr, side="left") - 1
        j = np.searchsorted(fomc, ts_arr, side="right")
        s["days_since_fomc"] = np.where(i >= 0, (ts_arr - fomc[np.clip(i, 0, None)]) / 86400, 60).clip(0, 60)
        s["days_to_fomc"] = np.where(j < len(fomc), (fomc[np.clip(j, None, len(fomc) - 1)] - ts_arr) / 86400, 60).clip(0, 60)
    else:
        s["days_since_fomc"], s["days_to_fomc"] = 60.0, 60.0
    return s


def news_features_sql(con, ref_ts: list[int]) -> pd.DataFrame:
    """Join reference timestamps to news IN SQL. Only items with an exact timestamp strictly BEFORE ref_ts."""
    con.execute("DROP TABLE IF EXISTS tmp_ref")
    con.execute("CREATE TABLE tmp_ref(ref_ts INTEGER PRIMARY KEY)")
    con.executemany("INSERT OR IGNORE INTO tmp_ref VALUES(?)", [(int(x),) for x in ref_ts])
    out = None
    for hrs in (24, 72):
        q = f"""
        SELECT r.ref_ts AS slot_ts,
               COUNT(n.news_id)       AS n_{hrs},
               AVG(n.sentiment)       AS sent_{hrs}h,
               AVG(n.policy_tone)     AS tone_{hrs}h,
               AVG(n.recession_flag)  AS recession_{hrs}h
        FROM tmp_ref r
        LEFT JOIN news n
               ON n.date_quality = 'exact'
              AND n.published_utc <  r.ref_ts
              AND n.published_utc >= r.ref_ts - {hrs * 3600}
        GROUP BY r.ref_ts"""
        d = pd.read_sql(q, con)
        out = d if out is None else out.merge(d, on="slot_ts")
    con.execute("DROP TABLE IF EXISTS tmp_ref")
    out["news_n_24h"] = np.log1p(out["n_24"])
    out["sent_delta"] = out["sent_24h"].fillna(0) - out["sent_72h"].fillna(0)
    cols = ["sent_24h", "sent_72h", "tone_24h", "recession_24h"]
    out[cols] = out[cols].fillna(0.0)
    return out[["slot_ts", "sent_24h", "sent_72h", "sent_delta", "news_n_24h", "tone_24h", "recession_24h"]]
