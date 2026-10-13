"""Sanity check of the whole modelling stack: when a real effect is injected into synthetic prices, walk-forward
validation must find it (positive OOS rank-IC, beats random picks). Guards against silently-broken plumbing."""
import pandas as pd
from macro import features as F, strategies as S, model as Mdl
from macro.backtest import evaluate
from macro.config import load_config
from macro.ingest.calendar_parse import normalize_calendar
from macro.ingest.synthetic import make_synthetic

NOW = pd.Timestamp("2026-10-13 12:00", tz="UTC")


def test_recovers_injected_signal():
    cfg = load_config()
    cfg["synthetic"] = {"signal_scale": 4.0}
    cfg["validation"].update(n_random_draws=100, bootstrap_reps=200)
    items, _, prices = make_synthetic(cfg, NOW)
    ev = F.enrich_events(normalize_calendar(items), int(NOW.timestamp()))
    slots = F.build_slots(ev, cfg["project"]["display_tz"])
    hist = slots[(slots.slot_ts >= int(NOW.timestamp()) - 180 * 86400) & (slots.released == 1) & (slots.slot_ts <= int(NOW.timestamp()))].copy()
    for c in Mdl.NEWS:
        hist[c] = 0.0
    rows = S.build_rows(hist, prices, cfg).merge(hist, on="slot_ts")
    oos, _, _ = Mdl.walk_forward(rows, cfg)
    res = evaluate(oos, prices, cfg)
    assert res["predictive"]["gbm"]["rank_ic_mean"] > 0.1
    assert res["stats"]["gbm"]["sharpe"] > 1.0
    assert res["perm_p_value"] < 0.1
