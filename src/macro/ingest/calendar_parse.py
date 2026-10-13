"""Normalise raw calendar rows (any Apify calendar actor) into one schema; UTC + US/Eastern alignment."""
from __future__ import annotations
import hashlib
import re
import numpy as np
import pandas as pd

_SCALE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

# (regex, category, polarity, tag).  polarity=+1: higher print = stronger economy / more hawkish.
RULES = [
    (r"fomc (meeting )?minutes", "policy", 1, "fomc_minutes"),
    (r"federal funds rate|fomc|interest rate decision|rate statement", "policy", 1, "fomc"),
    (r"fed chair|chair .* speaks|powell|fed .* speaks|fomc member", "policy", 1, "fed_speech"),
    (r"\bcore cpi\b|\bcpi\b", "inflation", 1, "cpi"),
    (r"\bcore pce\b|\bpce\b", "inflation", 1, "pce"),
    (r"\bppi\b", "inflation", 1, "ppi"),
    (r"average hourly earnings|employment cost", "inflation", 1, "wages"),
    (r"unemployment rate", "labor", -1, "nfp"),
    (r"unemployment claims|jobless", "labor", -1, "claims"),
    (r"non-?farm|\bnfp\b", "labor", 1, "nfp"),
    (r"\badp\b", "labor", 1, "adp"),
    (r"jolts|job openings", "labor", 1, "jolts"),
    (r"\bgdp\b", "growth", 1, "gdp"),
    (r"retail sales", "growth", 1, "retail"),
    (r"\bism\b|\bpmi\b|empire state|philly fed", "growth", 1, "pmi"),
    (r"industrial production|durable goods|factory orders", "growth", 1, "industrial"),
    (r"consumer (confidence|sentiment)|michigan", "growth", 1, "sentiment"),
    (r"housing|home sales|building permits|construction spending", "growth", 1, "housing"),
]
KEY_TAGS = ["cpi", "fomc", "nfp", "pce", "gdp", "pmi", "retail", "claims"]
CATS = ["inflation", "labor", "growth", "policy"]


def parse_value(x) -> float:
    """'0.5%'->0.5, '225K'->225000, '-0.1%'->-0.1, '3.5|2.1'->3.5, ''->nan"""
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", "").replace("<", "").replace(">", "")
    if s in ("", "-", "nan", "None", "null"):
        return np.nan
    s = s.split("|")[0]
    m = re.match(r"^([-+]?\d*\.?\d+)\s*([KMBT]?)", s, re.I)
    if not m:
        return np.nan
    v = float(m.group(1))
    return v * _SCALE.get(m.group(2).upper(), 1.0)


def classify_event(title: str) -> tuple[str, int, str]:
    t = (title or "").lower()
    for pat, cat, pol, tag in RULES:
        if re.search(pat, t):
            return cat, pol, tag
    return "other", 1, "other"


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def normalize_calendar(items: list[dict], display_tz: str = "America/New_York", currency: str = "USD") -> pd.DataFrame:
    rows = []
    for it in items:
        ts_raw = _first(it, ["datetimeISO", "datetime", "timestamp", "date_time", "date"])
        if ts_raw is None and "dateline" in it:
            ts = pd.to_datetime(int(it["dateline"]), unit="s", utc=True)
        else:
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
        if pd.isna(ts):
            continue
        title = str(_first(it, ["title", "event", "name", "prefixedName"], "")).strip()
        cur = str(_first(it, ["currency", "country"], "")).upper()
        impact = str(_first(it, ["impact", "importance"], "low")).lower()
        if currency and cur and cur != currency:
            continue
        if impact not in ("high", "medium"):
            continue
        cat, pol, tag = classify_event(title)
        rid = _first(it, ["id", "event_id"])
        eid = str(rid) if rid is not None else hashlib.md5(f"{ts}{title}".encode()).hexdigest()[:12]
        rows.append({
            "event_id": eid, "ts_utc": int(ts.timestamp()),
            "ts_et": ts.tz_convert(display_tz).isoformat(), "title": title, "currency": cur or currency,
            "impact": impact, "category": cat, "polarity": pol, "tag": tag,
            "actual": parse_value(it.get("actual")), "forecast": parse_value(it.get("forecast")),
            "previous": parse_value(it.get("previous")), "revision": parse_value(it.get("revision")),
            "actual_raw": it.get("actual"), "forecast_raw": it.get("forecast"), "previous_raw": it.get("previous"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates("event_id").sort_values(["ts_utc", "title"]).reset_index(drop=True)
