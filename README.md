# Macro Event → Strategy P&L Predictor
CrowdWisdomTrading — Quantitative Data Scientist intern assessment

A Python pipeline that scrapes a US macro calendar (Apify), pulls macro news (Tavily, optional Exa), joins everything in
SQLite, engineers time/surprise/sentiment features, and trains a model that predicts the **net P&L / win-rate of every
strategy permutation after each release**. It is validated with a **purged, expanding-window walk-forward** and reports
**Sharpe, Sortino and Max Drawdown**, plus charts and a one-page macro outlook.

```
 Apify actor ──► events ─┐
 Tavily/Exa  ──► news  ──┼─► SQLite (events, news, prices, slots) ─► features ─► labels (36 permutations / release)
 yfinance    ──► prices ─┘                                              │
                                                                         ▼
        reports/ ◄── charts, outlook, evaluation report ◄── walk-forward (Ridge / GBM / P(win) classifier)
```

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # or: pip install -r requirements.txt  (+ export PYTHONPATH=src)
pytest -q                             # 13 tests, no network needed

python -m macro demo                  # fully offline run on SYNTHETIC data -> reports/demo/
cp .env.example .env                  # add APIFY_TOKEN and TAVILY_API_KEY
python -m macro live                  # real run -> reports/evaluation_report.md, reports/macro_outlook.md, reports/figures/
```
Sub-commands: `ingest | build | train | report` (add `--mode demo|live`, `--refresh` to bypass cached Apify runs,
`--llm` to polish the outlook with Claude if `ANTHROPIC_API_KEY` is set).

## What is scraped (reproducibility)
| Source | ID / endpoint | Used for |
|---|---|---|
| Apify actor | **`xtracto/forexfactory-calendar`** (input: custom date range, `currencies:[USD]`, `minImpact:medium`, residential proxy) | calendar: timestamp (UTC + US/Eastern), event, actual, forecast, previous, impact |
| Tavily | `POST https://api.tavily.com/search` (`topic: news`, weekly windows + latest week) | news + exact `published_date` |
| Exa (optional) | `POST https://api.exa.ai/search` (`category: news`) | second news source, exact publish dates |
| yfinance | hourly bars: ES=F, NQ=F, ZN=F, GC=F, EURUSD=X, CL=F | strategy P&L labels |

Every Apify run (actor id, run id, dataset id, window, input) is written to `data/raw/manifest.json`, and raw payloads are
cached under `data/raw/` — so a reviewer can re-run `python -m macro live` **without a token** from the cached files.
Swap or add actors in `config.yaml → apify.actors` (fallbacks are tried in order).

## Design
* **Decision unit = release slot.** CPI m/m + Core CPI + CPI y/y at the same timestamp are *one* trade decision (no triple counting).
* **Permutations:** 6 assets × long/short × 1h/4h/24h hold = 36 per release. Entry = open of the first hourly bar after release (+1 min latency).
* **Target:** vol-scaled net P&L in bps of capital after per-asset costs (+ binary win label for a P(win) classifier).
* **Features** (all known at entry): cyclical hour/day-of-week, minutes-between-releases, days since/to FOMC (scheduled ⇒ public),
  release-type flags (CPI/FOMC/NFP/PCE/GDP/PMI/retail/claims), **causal surprise z-scores** per theme (inflation/labor/growth/policy,
  polarity-adjusted so + = hawkish/strong), 24h/72h news sentiment, sentiment change, policy tone, recession-theme share, and
  per-asset trailing 4h/24h return + vol, plus direction×signal interactions.
* **Join in the database:** events ⟷ news is a SQL `LEFT JOIN` on `published_utc < release_ts` (`features.news_features_sql`).
* **Models:** Ridge baseline, `HistGradientBoostingRegressor` (**pre-specified primary**), GBM classifier for P(win). Fixed hyper-parameters in `config.yaml` — nothing is tuned on test folds.
* **Validation:** 5 expanding-window folds; training rows must have *exited* ≥24h before the first test entry (purge + embargo).
* **Reported:** Sharpe, Sortino, Max DD, Calmar, win rate, profit factor, rank-IC, AUC, per-fold stability, block-bootstrap Sharpe CI, and a
  **random-pick permutation test** (is the model better than choosing a permutation at random?). Benchmarks: naive long-ES and buy&hold ES.

## Outputs
`reports/evaluation_report.md` · `reports/macro_outlook.md` (1-page) · `reports/figures/`:
`sentiment_timeline` (sentiment & policy tone vs CPI/FOMC/NFP markers), `topic_sentiment_matrix` (topic × week), `surprise_matrix`
(theme × week), `scenario_matrix` (predicted P&L for the next key release: below / in-line / above consensus), `equity_curve`,
`walkforward_folds`, `pnl_matrix` (asset × horizon), `feature_importance`.

## Layout
```
config.yaml            all knobs (window, actors, queries, assets, costs, model, validation)
src/macro/ingest/      apify_calendar.py · news.py (Tavily/Exa) · prices.py · calendar_parse.py · synthetic.py (offline demo only)
src/macro/             db.py · features.py · strategies.py · model.py · backtest.py · metrics.py · viz.py · summary.py · report.py · pipeline.py
tests/                 parsing, metric formulas, leakage tests, walk-forward purge, signal-recovery, mocked Apify/Tavily
reports/demo/          output of `python -m macro demo` (SYNTHETIC — not real performance)
```

## Pipeline Status & Reproduction
* **Live Pipeline Completed:** The pipeline has been executed against live services:
  - Macro calendar scraped via Apify actor `xtracto/forexfactory-calendar` (10 windows covering the 180-day window + 365-day surprise warmup, 830 real events). All run IDs and dataset IDs are logged in `data/raw/manifest.json`.
  - Macroeconomic news ingested with exact timestamps and sentiment scored via VADER.
  - Hourly bars for 6 macro futures/currencies (ES, NQ, ZN, GC, EURUSD, CL) fetched via `yfinance` (21,858 bars).
* **Live Results:** The real evaluation report is in `reports/evaluation_report.md`, with 8 high-resolution charts in `reports/figures/` and the 1-page summary in `reports/macro_outlook.md`.
* **Zero-Token Reproduction:** All raw Apify scraped JSON payloads are committed under `data/raw/apify/`, allowing reviewers to reproduce `python -m macro live` immediately without needing an API token or consuming Apify credits.
* **Offline Demo Mode:** `python -m macro demo` remains available for CI/sandbox testing with controlled synthetic data in `reports/demo/`.
