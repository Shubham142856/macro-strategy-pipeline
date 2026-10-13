import numpy as np
import pandas as pd
from macro.ingest.calendar_parse import parse_value, classify_event, normalize_calendar
from macro import metrics as M
from macro.sentiment import score_text


def test_parse_value():
    assert parse_value("0.5%") == 0.5 and parse_value("-0.1%") == -0.1
    assert parse_value("225K") == 225_000 and parse_value("1.2M") == 1_200_000
    assert parse_value("3.5|2.1") == 3.5
    assert np.isnan(parse_value("")) and np.isnan(parse_value(None)) and np.isnan(parse_value("abc"))


def test_classify():
    assert classify_event("Core CPI m/m")[2] == "cpi"
    assert classify_event("Federal Funds Rate")[2] == "fomc"
    assert classify_event("FOMC Meeting Minutes")[2] == "fomc_minutes"
    assert classify_event("Unemployment Claims")[1] == -1          # higher claims = weaker economy
    assert classify_event("Non-Farm Employment Change")[2] == "nfp"


def test_normalize_utc_and_et():
    it = [{"id": 1, "title": "CPI m/m", "currency": "USD", "impact": "high", "datetimeISO": "2026-07-14T12:30:00Z",
           "actual": "0.3%", "forecast": "0.2%", "previous": "0.1%"},
          {"id": 2, "title": "x", "currency": "EUR", "impact": "high", "datetimeISO": "2026-07-14T12:30:00Z"},
          {"id": 3, "title": "y", "currency": "USD", "impact": "low", "datetimeISO": "2026-07-14T12:30:00Z"}]
    df = normalize_calendar(it)
    assert len(df) == 1 and df.ts_et.iloc[0].startswith("2026-07-14T08:30:00-04:00")


def test_metrics_formulas():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    eq = (1 + r).cumprod()
    assert abs(M.max_drawdown(r) - float((eq / eq.cummax() - 1).min())) < 1e-12
    s = M.perf_stats(r)
    assert abs(s["sharpe"] - r.mean() / r.std(ddof=1) * np.sqrt(252)) < 1e-9
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert abs(s["sortino"] - r.mean() / dd * np.sqrt(252)) < 1e-9
    assert M.max_drawdown(pd.Series([0.1, 0.1, 0.1])) == 0.0


def test_sentiment_negation():
    assert score_text("Stocks rally on strong data")["sentiment"] > 0
    assert score_text("Recession fears and weak growth")["sentiment"] < 0
    assert score_text("not weak at all")["sentiment"] >= score_text("weak")["sentiment"]
    assert score_text("Fed hawkish, higher for longer")["policy_tone"] > 0
