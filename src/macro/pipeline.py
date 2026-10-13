from __future__ import annotations
import json
import logging
import numpy as np
import pandas as pd
from . import db, features as F, model as Mdl, strategies as S, viz, summary as Sum, report as Rep
from .backtest import evaluate
from .config import ROOT, data_path, load_config, sym_short
from .ingest.calendar_parse import KEY_TAGS, normalize_calendar
from .ingest.news import normalize_news

log = logging.getLogger(__name__)


def cfg_for(mode: str) -> dict:
    """Demo runs use their own DB + output folder so synthetic numbers can never overwrite live results."""
    cfg = load_config()
    if mode == "demo":
        cfg["project"]["db_path"] = "data/macro_demo.db"
    return cfg


def out_dir(mode: str):
    d = ROOT / "reports" / ("demo" if mode == "demo" else "")
    (d / "figures").mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------ 1. ingest
def ingest(cfg: dict, mode: str, refresh: bool = False) -> None:
    con = db.connect(cfg)
    now = pd.Timestamp.now(tz="UTC").floor("h")
    lb, wu = cfg["project"]["lookback_days"], cfg["project"]["warmup_days"]
    if mode == "demo":
        from .ingest.synthetic import make_synthetic
        items, news_items, prices = make_synthetic(cfg, now)
        ev = normalize_calendar(items, cfg["project"]["display_tz"])
        news = normalize_news(news_items)
        man = []
    else:
        from .ingest.apify_calendar import fetch_calendar
        from .ingest.news import fetch_news
        from .ingest.prices import fetch_prices
        ev = fetch_calendar(cfg, now - pd.Timedelta(days=lb + wu), now + pd.Timedelta(days=14), refresh)
        news = fetch_news(cfg, now - pd.Timedelta(days=lb), now)
        prices = fetch_prices(cfg, now - pd.Timedelta(days=lb + 45), now)
        mp = data_path("data", "raw", "manifest.json")
        man = json.loads(mp.read_text()) if mp.exists() else []
    db.write_table(con, "events", ev, ["ts_utc"])
    db.write_table(con, "news", news, ["published_utc"])
    db.write_table(con, "prices", prices, ["symbol", "ts_utc"])
    db.set_meta(con, "mode", mode)
    db.set_meta(con, "now_ts", int(now.timestamp()))
    db.set_meta(con, "manifest", json.dumps(man))
    log.info("ingested: %d events, %d news, %d bars (%s)", len(ev), len(news), len(prices), mode)


# ------------------------------------------------------------------ 2. features + labels
def build(cfg: dict) -> None:
    con = db.connect(cfg)
    now_ts = int(db.get_meta(con, "now_ts"))
    tz = cfg["project"]["display_tz"]
    ev = F.enrich_events(db.read_table(con, "events"), now_ts)
    db.write_table(con, "events", ev, ["ts_utc"])
    slots_all = F.build_slots(ev, tz)
    start = now_ts - cfg["project"]["lookback_days"] * 86400
    hist = slots_all[(slots_all.slot_ts >= start) & (slots_all.slot_ts <= now_ts) & (slots_all.released == 1)].copy()
    hist = hist.merge(F.news_features_sql(con, hist.slot_ts.tolist()), on="slot_ts", how="left")
    up = slots_all[slots_all.slot_ts > now_ts].copy()
    prices = db.read_table(con, "prices")
    rows = S.build_rows(hist, prices, cfg).merge(hist, on="slot_ts", how="left")
    db.write_table(con, "slots", hist, ["slot_ts"])
    db.write_table(con, "slots_upcoming", up, ["slot_ts"])
    db.write_table(con, "permutation_rows", rows, ["slot_ts"])
    log.info("slots=%d (traded=%d) rows=%d upcoming=%d", len(slots_all), hist.slot_ts.nunique(), len(rows), len(up))


# ------------------------------------------------------------------ 3. train + validate
def _scenario(ms, up: pd.DataFrame, con, cfg: dict) -> pd.DataFrame:
    now_ts = int(db.get_meta(con, "now_ts"))
    cand = up[(up[[f"is_{t}" for t in ("cpi", "fomc", "nfp", "pce", "gdp", "retail")]].sum(axis=1) > 0)]
    if cand.empty:
        cand = up[up[[f"is_{t}" for t in KEY_TAGS]].sum(axis=1) > 0]
    if cand.empty:
        return pd.DataFrame()
    slot = cand.iloc[0]
    cat = "policy" if slot.is_fomc else "inflation" if (slot.is_cpi or slot.is_pce) else "labor" if slot.is_nfp else "growth"
    news_now = F.news_features_sql(con, [now_ts]).iloc[0]
    state = S.market_state_now(db.read_table(con, "prices"), cfg)
    h = 4 if 4 in cfg["strategy"]["horizons_bars"] else cfg["strategy"]["horizons_bars"][0]
    recs = []
    for name, z in [("Below consensus", -1.5), ("In line", 0.0), ("Above consensus", 1.5)]:
        for _, a in state.iterrows():
            for d in cfg["strategy"]["directions"]:
                r = {**{c: slot.get(c, 0.0) for c in Mdl.BASE + Mdl.FLAGS + Mdl.SURP}, **{c: news_now[c] for c in Mdl.NEWS},
                     "pre_ret_4h": a.pre_ret_4h, "pre_ret_24h": a.pre_ret_24h, "pre_vol_bps": a.pre_vol_bps,
                     "asset": a.asset, "direction": d, "horizon": h, "scenario": name, "headline": slot.headline, "slot_ts": slot.slot_ts}
                r[f"s_{cat}"] = z
                recs.append(r)
    df = pd.DataFrame(recs)
    pr = Mdl.predict(ms, Mdl.assemble(df, cfg))
    return pd.concat([df[["scenario", "headline", "slot_ts", "asset", "direction", "horizon"]], pr], axis=1)


def train(cfg: dict) -> None:
    con = db.connect(cfg)
    rows = db.read_table(con, "permutation_rows")
    oos, info, fi = Mdl.walk_forward(rows, cfg)
    ms = Mdl.fit_final(rows, cfg)
    scen = _scenario(ms, db.read_table(con, "slots_upcoming"), con, cfg)
    db.write_table(con, "oos_predictions", oos)
    info2 = info.copy()
    for c in ("train_end_exit_utc", "test_start_utc", "test_end_utc"):
        info2[c] = info2[c].astype(str)
    db.write_table(con, "wf_folds", info2)
    db.write_table(con, "feature_importance", fi.rename("importance").rename_axis("feature").reset_index())
    db.write_table(con, "scenario_matrix", scen if len(scen) else pd.DataFrame({"scenario": []}))
    log.info("walk-forward done: %d OOS rows over %d folds", len(oos), len(info))


# ------------------------------------------------------------------ 4. charts + reports
def report(cfg: dict, llm: bool = False) -> None:
    con = db.connect(cfg)
    mode = db.get_meta(con, "mode")
    OUT = out_dir(mode)
    FIG = OUT / "figures"
    now_ts = int(db.get_meta(con, "now_ts"))
    tz = cfg["project"]["display_tz"]
    ev, news, prices = db.read_table(con, "events"), db.read_table(con, "news"), db.read_table(con, "prices")
    slots, up = db.read_table(con, "slots"), db.read_table(con, "slots_upcoming")
    oos = db.read_table(con, "oos_predictions")
    fi = db.read_table(con, "feature_importance").set_index("feature")["importance"]
    scen = db.read_table(con, "scenario_matrix")
    scen = scen if "pred_gbm" in scen.columns else pd.DataFrame()
    fold_info = db.read_table(con, "wf_folds")
    for c in ("train_end_exit_utc", "test_start_utc", "test_end_utc"):
        fold_info[c] = pd.to_datetime(fold_info[c])
    res = evaluate(oos, prices, cfg)

    viz.fig_sentiment_timeline(news, slots, FIG / "sentiment_timeline.png")
    viz.fig_topic_matrix(news, FIG / "topic_sentiment_matrix.png")
    viz.fig_surprise_matrix(ev, FIG / "surprise_matrix.png")
    viz.fig_scenario(scen, FIG / "scenario_matrix.png")
    viz.fig_equity(res, FIG / "equity_curve.png")
    viz.fig_folds(res, fold_info, FIG / "walkforward_folds.png")
    viz.fig_pnl_matrix(oos, FIG / "pnl_matrix.png")
    viz.fig_importance(fi, FIG / "feature_importance.png")

    text = Sum.build_summary(ev, news, up, scen, now_ts, tz, mode == "demo")
    if llm:
        text = Sum.llm_polish(text)
    (OUT / "macro_outlook.md").write_text(text, encoding="utf-8")

    t0, t1 = res["window"]
    prov = {"mode": mode, "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
            "oos_start": pd.to_datetime(t0, unit="s").strftime("%Y-%m-%d"), "oos_end": pd.to_datetime(t1, unit="s").strftime("%Y-%m-%d"),
            "n_events": len(ev), "n_slots": int(slots.slot_ts.nunique()), "n_news": len(news), "n_news_exact": int((news.date_quality == "exact").sum()),
            "n_bars": len(prices), "exa": bool((news.source_api == "exa").any()), "manifest": json.loads(db.get_meta(con, "manifest", "[]"))}
    Rep.write_report(res, oos, fold_info, fi, prov, cfg, OUT / "evaluation_report.md")
    s = res["stats"][res["primary"]]
    log.info("PRIMARY (%s): Sharpe=%.2f Sortino=%.2f MaxDD=%.1f%% trades=%d perm-p=%.3f", res["primary"], s["sharpe"], s["sortino"],
             s["max_drawdown"] * 100, s["n_trades"], res["perm_p_value"])
    return res


def run_all(mode: str, refresh: bool = False, llm: bool = False):
    cfg = cfg_for(mode)
    ingest(cfg, mode, refresh)
    build(cfg)
    train(cfg)
    return report(cfg, llm)
