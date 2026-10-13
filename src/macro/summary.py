"""One-page macro outlook (template-driven from the data; optional LLM polish via ANTHROPIC_API_KEY)."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import requests

CAT_WORD = {"inflation": ("hotter", "cooler"), "labor": ("stronger", "weaker"), "growth": ("stronger", "weaker"),
            "policy": ("more hawkish", "more dovish")}


def _et(ts, tz):
    return pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz)


def build_summary(ev: pd.DataFrame, news: pd.DataFrame, slots_up: pd.DataFrame, scen: pd.DataFrame, now_ts: int,
                  tz: str, demo: bool) -> str:
    now = _et(now_ts, tz)
    n = news[news.date_quality == "exact"].copy()
    n["t"] = pd.to_datetime(n.published_utc, unit="s", utc=True)
    s7 = n[n.t >= pd.to_datetime(now_ts - 7 * 86400, unit="s", utc=True)]
    s30 = n[n.t >= pd.to_datetime(now_ts - 30 * 86400, unit="s", utc=True)]
    sent7, sent30 = s7.sentiment.mean(), s30.sentiment.mean()
    tone7 = s7.policy_tone.mean()
    rec7 = s7.recession_flag.mean() * 100
    trend = "improving" if sent7 - sent30 > 0.05 else ("deteriorating" if sent7 - sent30 < -0.05 else "broadly stable")
    tone_w = "hawkish-leaning" if tone7 > 0.1 else ("dovish-leaning" if tone7 < -0.1 else "neutral")
    L = []
    if demo:
        L.append("> **SYNTHETIC DEMO DATA — illustrative only. Re-run with live Apify/Tavily data for the real outlook.**\n")
    L.append(f"# Macro Outlook — {now:%d %b %Y} (US/Eastern)\n")
    L.append(f"**Regime.** 7-day news sentiment is **{sent7:+.2f}** (30-day {sent30:+.2f}) → risk appetite is **{trend}**; policy tone is "
             f"**{tone_w}** ({tone7:+.2f}); {rec7:.0f}% of last-week articles mention recession/stagflation themes.\n")
    e = ev[(ev.released == 1) & ev.signed_score.notna() & (ev.ts_utc >= now_ts - 30 * 86400)]
    L.append("**Data surprises (last 30 days, mean z-score vs consensus).**\n")
    L.append("| Theme | Releases | Mean surprise z | Read |\n|---|---:|---:|---|")
    for c in ["inflation", "labor", "growth", "policy"]:
        x = e[e.category == c].signed_score
        if len(x):
            hi, lo = CAT_WORD[c]
            read = hi if x.mean() > 0.25 else (lo if x.mean() < -0.25 else "in line")
            L.append(f"| {c.title()} | {len(x)} | {x.mean():+.2f} | {read} than consensus |" if c != "policy" else f"| Policy | {len(x)} | {x.mean():+.2f} | {read} |")
    L.append("")
    key = ev[(ev.released == 1) & ev.tag.isin(["cpi", "nfp", "pce", "fomc", "gdp"]) & ev.actual.notna()].sort_values("ts_utc").tail(6)
    L.append("**Latest key releases.**\n")
    L.append("| Date (ET) | Release | Actual | Forecast | Previous |\n|---|---|---:|---:|---:|")
    for _, r in key.iloc[::-1].iterrows():
        L.append(f"| {_et(r.ts_utc, tz):%d %b %H:%M} | {r.title} | {r.actual_raw} | {r.forecast_raw} | {r.previous_raw} |")
    L.append("")
    if slots_up is not None and len(slots_up):
        u = slots_up[(slots_up.slot_ts > now_ts) & (slots_up.slot_ts <= now_ts + 10 * 86400) &
                     (slots_up[["is_cpi", "is_fomc", "is_nfp", "is_pce", "is_gdp", "is_retail", "is_pmi"]].sum(axis=1) > 0)].head(6)
        L.append("**Upcoming catalysts (next 10 days).** " + "; ".join(f"{_et(r.slot_ts, tz):%a %d %b %H:%M} {r.headline}" for _, r in u.iterrows()) + "\n")
    if scen is not None and len(scen):
        L.append(f"**Model read-through for *{scen.headline.iloc[0]}* (GBM, {scen.horizon.iloc[0]}h hold, predicted net bps of capital).**\n")
        for sc in ["Below consensus", "In line", "Above consensus"]:
            g = scen[scen.scenario == sc].sort_values("pred_gbm", ascending=False).head(2)
            L.append(f"- *{sc}:* " + ", ".join(f"{r.asset} {'long' if r.direction > 0 else 'short'} ({r.pred_gbm:+.1f})" for _, r in g.iterrows()))
        L.append("")
    heads = n.sort_values("published_utc").tail(4)
    if len(heads):
        L.append("**Latest headlines scanned.** " + " | ".join(h.title[:90] for h in heads.itertuples()) + "\n")
    L.append("*Outlook statistics are rule-based aggregates of scraped calendar + news data; model figures are statistical "
             "forecasts, not investment advice.*")
    return "\n".join(L)


def llm_polish(text: str, model: str | None = None) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return text
    r = requests.post("https://api.anthropic.com/v1/messages", timeout=90,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"), "max_tokens": 900,
                            "messages": [{"role": "user", "content": "Rewrite this data-derived macro brief as a crisp one-page "
                                          "macro outlook for a trading desk. Keep every number and table unchanged, add no new "
                                          "facts, keep it under 450 words.\n\n" + text}]})
    if r.status_code != 200:
        return text
    return "".join(b.get("text", "") for b in r.json().get("content", [])) or text
