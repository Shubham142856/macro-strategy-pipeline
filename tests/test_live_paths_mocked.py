"""Live ingestion plumbing exercised with mocked network (no tokens needed)."""
import json
import pandas as pd
import macro.config as C
from macro.config import load_config

FAKE_ROWS = [
    {"id": 1, "title": "CPI m/m", "currency": "USD", "impact": "high", "datetimeISO": "2026-07-14T12:30:00Z",
     "actual": "0.3%", "forecast": "0.2%", "previous": "0.1%", "revision": ""},
    {"id": 2, "title": "Federal Funds Rate", "currency": "USD", "impact": "high", "datetimeISO": "2026-07-29T18:00:00Z",
     "actual": "4.25%", "forecast": "4.25%", "previous": "4.25%", "revision": ""},
]


def test_apify_fetch_and_manifest(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setenv("APIFY_TOKEN", "x")
    calls = []

    class FakeDataset:
        def iterate_items(self):
            return iter(FAKE_ROWS)

    class FakeActor:
        def __init__(self, a): self.a = a
        def call(self, run_input, timeout_secs):
            calls.append((self.a, run_input))
            return {"id": "RUN1", "defaultDatasetId": "DS1"}

    class FakeClient:
        def __init__(self, token): pass
        def actor(self, a): return FakeActor(a)
        def dataset(self, d): return FakeDataset()

    import apify_client
    monkeypatch.setattr(apify_client, "ApifyClient", FakeClient)
    from macro.ingest.apify_calendar import fetch_calendar
    df = fetch_calendar(cfg, pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-07-31", tz="UTC"))
    assert set(df.title) == {"CPI m/m", "Federal Funds Rate"}
    actor, inp = calls[0]
    assert actor == "xtracto/forexfactory-calendar" and inp["rangeFrom"] == "2026-07-01" and inp["rangeTo"] == "2026-07-31"
    man = json.loads((tmp_path / "data" / "raw" / "manifest.json").read_text())
    assert man[0]["run_id"] == "RUN1" and man[0]["actor_id"] == "xtracto/forexfactory-calendar"
    # second call is served from cache (reproducible without a token)
    calls.clear()
    fetch_calendar(cfg, pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-07-31", tz="UTC"))
    assert calls == []


def test_tavily_normalisation_and_dates(tmp_path, monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    import macro.ingest.news as N

    class R:
        status_code = 200
        def json(self): return {"results": [
            {"title": "Stocks rally as CPI cools", "url": "https://a.com/1", "content": "Inflation cools; rate cut hopes.",
             "published_date": "Tue, 14 Jul 2026 13:05:00 GMT"},
            {"title": "Undated macro piece", "url": "https://a.com/2", "content": "Recession fears", "published_date": None}]}
    monkeypatch.setattr(N.requests, "post", lambda *a, **k: R())
    cfg["tavily"]["queries"] = cfg["tavily"]["queries"][:1]
    df = N.fetch_news(cfg, pd.Timestamp("2026-07-10", tz="UTC"), pd.Timestamp("2026-07-17", tz="UTC"))
    assert (df.date_quality == "exact").sum() == 1 and (df.date_quality == "missing").sum() == 1
    assert df.loc[df.date_quality == "exact", "sentiment"].iloc[0] > 0
