"""Leakage / validation-integrity tests on a small synthetic run (no network)."""
import numpy as np
import pandas as pd
import pytest
from macro import features as F, strategies as S, model as Mdl, db
from macro.config import load_config
from macro.ingest.calendar_parse import normalize_calendar
from macro.ingest.news import normalize_news
from macro.ingest.synthetic import make_synthetic

NOW = pd.Timestamp("2026-10-13 12:00", tz="UTC")


@pytest.fixture(scope="module")
def world():
    cfg = load_config()
    items, news_items, prices = make_synthetic(cfg, NOW)
    ev = F.enrich_events(normalize_calendar(items), int(NOW.timestamp()))
    news = normalize_news(news_items)
    slots = F.build_slots(ev, cfg["project"]["display_tz"])
    start = int(NOW.timestamp()) - 180 * 86400
    hist = slots[(slots.slot_ts >= start) & (slots.released == 1) & (slots.slot_ts <= int(NOW.timestamp()))]
    return cfg, ev, news, prices, hist


def test_entry_strictly_after_release(world):
    cfg, ev, news, prices, hist = world
    rows = S.build_rows(hist, prices, cfg)
    assert len(rows) > 500
    assert (rows.entry_ts >= rows.slot_ts + 60).all()           # entry bar starts after release + latency
    assert (rows.exit_ts > rows.entry_ts).all()


def test_features_do_not_see_the_future(world):
    """Scramble every price bar AFTER entry and every news item AFTER the release: entry-time features must not move."""
    cfg, ev, news, prices, hist = world
    rng = np.random.default_rng(0)
    base = S.build_rows(hist, prices, cfg)
    p2 = prices.copy()
    cut = int(hist.slot_ts.iloc[len(hist) // 2])
    mask = p2.ts_utc > cut + 3600
    p2.loc[mask, ["open", "high", "low", "close"]] *= rng.uniform(0.9, 1.1, mask.sum())[:, None]
    after = S.build_rows(hist[hist.slot_ts <= cut], p2, cfg)
    b = base[base.slot_ts <= cut].reset_index(drop=True)
    a = after.reset_index(drop=True)
    cols = ["slot_ts", "asset", "direction", "horizon", "pre_ret_4h", "pre_ret_24h", "pre_vol_bps"]
    pd.testing.assert_frame_equal(b[cols], a[cols])


def test_news_join_ignores_future_news(world, tmp_path, monkeypatch):
    cfg, ev, news, prices, hist = world
    cfg = {**cfg, "project": {**cfg["project"], "db_path": str(tmp_path / "t.db")}}
    import macro.db as dbm
    from macro import config as C
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setattr(dbm, "ROOT", tmp_path)
    con = dbm.connect(cfg)
    ref = [int(hist.slot_ts.iloc[10])]
    dbm.write_table(con, "news", news)
    f1 = F.news_features_sql(con, ref)
    n2 = news.copy()
    fut = n2.published_utc >= ref[0]
    n2.loc[fut, "sentiment"] = 1.0                                   # poison everything at/after the release
    dbm.write_table(con, "news", n2)
    f2 = F.news_features_sql(con, ref)
    pd.testing.assert_frame_equal(f1, f2)


def test_surprise_std_is_causal(world):
    cfg, ev, *_ = world
    e = ev[ev.title == "CPI m/m"].sort_values("ts_utc")
    # std at row i must equal std of surprises strictly before i
    i = 8
    manual = e.surprise.iloc[:i].std()
    assert abs(e.sur_std.iloc[i] - manual) < 1e-9


def test_walk_forward_purge(world):
    cfg, ev, news, prices, hist = world
    hist = hist.copy()
    for c in ["sent_24h", "sent_72h", "sent_delta", "news_n_24h", "tone_24h", "recession_24h"]:
        hist[c] = 0.0
    rows = S.build_rows(hist, prices, cfg).merge(hist, on="slot_ts")
    cfg2 = {**cfg, "validation": {**cfg["validation"], "n_folds": 3}}
    oos, info, fi = Mdl.walk_forward(rows, cfg2)
    emb = cfg["validation"]["embargo_hours"] * 3600
    for _, r in info.iterrows():
        assert r.train_end_exit_utc.timestamp() <= r.test_start_utc.timestamp() - emb
    # folds are chronological and non-overlapping
    spans = oos.groupby("fold").slot_ts.agg(["min", "max"]).values
    assert all(spans[i, 1] < spans[i + 1, 0] for i in range(len(spans) - 1))
    # every OOS row is predicted exactly once
    assert not oos.duplicated(["slot_ts", "asset", "direction", "horizon"]).any()
