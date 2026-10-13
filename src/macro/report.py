"""Evaluation report (Markdown) — Sharpe, Sortino, Max Drawdown + validation design + honest limitations."""
from __future__ import annotations
import json
import pandas as pd
from .config import ROOT


def _t(rows: list[list], head: list[str]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


def write_report(res: dict, oos: pd.DataFrame, fold_info: pd.DataFrame, fi: pd.Series, prov: dict, cfg: dict, path) -> None:
    st, q, pr = res["stats"], res["predictive"], res["primary"]
    names = {"gbm": "GBM regressor (PRIMARY, pre-specified)", "ridge": "Ridge (linear baseline)", "gbm_gated": "GBM + P(win)≥0.5 gate",
             "naive_long_ES_4h": "Naive: long ES 4h on every release", "buy_hold_ES": "Buy & hold ES (benchmark)"}
    rows = []
    for k, n in names.items():
        s = st[k]
        rows.append([n, f"{s['sharpe']:.2f}", f"{s['sortino']:.2f}", f"{s['max_drawdown']*100:.1f}%", f"{s['ann_return']*100:.1f}%",
                     f"{s['calmar']:.2f}", s.get("n_trades", ""), f"{s['win_rate']*100:.0f}%" if "win_rate" in s else "—",
                     f"{s['profit_factor']:.2f}" if "profit_factor" in s else "—"])
    perf = _t(rows, ["Strategy", "Sharpe", "Sortino", "Max DD", "Ann. return", "Calmar", "Trades", "Win rate", "Profit factor"])
    ci = st[pr].get("sharpe_ci95", (float("nan"),) * 2)
    fold_rows = [[int(r.fold), r.test_start_utc.strftime("%Y-%m-%d"), r.test_end_utc.strftime("%Y-%m-%d"), int(r.train_rows), int(r.test_rows)] for _, r in fold_info.iterrows()]
    f = res["folds"]
    fold_perf = _t([[int(r.fold), int(r.n_trades), f"{r.sharpe:.2f}", f"{r.total_return_pct:.2f}%", f"{r.rank_ic:+.3f}"] for _, r in f.iterrows()],
                   ["Fold", "Trades", "Sharpe", "Return", "Rank IC"])
    pq = _t([[k, f"{v['rank_ic_mean']:+.3f}", f"{v['rank_ic_t']:.2f}", f"{v['ic_hit_rate']*100:.0f}%", f"{v['pooled_corr']:+.3f}", f"{v['r2']:+.4f}"]
             for k, v in q.items() if k != "classifier"], ["Model", "Mean rank IC", "IC t-stat", "IC>0 share", "Pooled corr", "OOS R²"])
    clf = q["classifier"]
    banner = ("> ⚠️ **SYNTHETIC DEMO RUN.** Calendar, news and prices in this report were *generated* by `macro.ingest.synthetic` to exercise "
              "the pipeline offline. Results demonstrate the machinery, **not** real market performance. Re-run with `python -m macro live`.\n\n"
              if prov["mode"] == "demo" else "")
    actors = "\n".join(f"- `{m['actor_id']}` run `{m.get('run_id')}` dataset `{m.get('dataset_id')}` ({m['from']} → {m['to']}, {m['n_items']} rows)" for m in prov.get("manifest", [])) or "- (synthetic mode: no Apify run)"
    md = f"""# Evaluation Report — Macro Event Strategy Predictor

{banner}**Data mode:** `{prov['mode']}` · **Generated:** {prov['generated']} · **OOS window:** {prov['oos_start']} → {prov['oos_end']}

## 1. Data & provenance
| Dataset | Rows | Source |
|---|---:|---|
| Macro events (USD, medium/high impact; incl. {cfg['project']['warmup_days']}d warm-up for surprise normalisation) | {prov['n_events']} | Apify `{cfg['apify']['actors'][0]['actor_id']}` |
| Release slots traded (one per release timestamp) | {prov['n_slots']} | derived |
| News articles (exact-timestamp only used as features) | {prov['n_news']} ({prov['n_news_exact']} exact) | Tavily{' + Exa' if prov.get('exa') else ''} |
| Hourly price bars ({len(cfg['markets']['assets'])} assets) | {prov['n_bars']} | yfinance |

Apify runs:
{actors}

## 2. Problem definition
* **Decision unit:** each release timestamp (CPI m/m + Core CPI + CPI y/y at 08:30 ET is *one* slot).
* **Strategy permutations per slot:** {len(cfg['markets']['assets'])} assets × {len(cfg['strategy']['directions'])} directions × {len(cfg['strategy']['horizons_bars'])} horizons ({cfg['strategy']['horizons_bars']} h) = **{len(cfg['markets']['assets'])*2*len(cfg['strategy']['horizons_bars'])}**.
* **Entry:** open of the first hourly bar starting ≥ release + {cfg['strategy']['entry_latency_minutes']} min (conservative: no fill inside the release bar). **Exit:** open of the bar `h` bars later.
* **Target:** net P&L in bps of capital, volatility-scaled so a 1σ move over the hold = {cfg['strategy']['risk_per_trade_bps']} bps (leverage cap {cfg['strategy']['max_leverage']}×), after per-asset round-trip costs. Also a binary win label (P&L>0).
* **Selection rule:** per release, trade the permutation with the highest predicted net P&L if it is > {cfg['model']['trade_threshold_bps']} bps (one trade per release).

## 3. Validation protocol
* **Expanding-window walk-forward** over release slots, {cfg['validation']['n_folds']} test folds, first {int(cfg['validation']['initial_train_frac']*100)}% of slots used only for initial training.
* **Purge + embargo:** training rows must have *exited* ≥ {cfg['validation']['embargo_hours']} h before the first test entry (no label overlap).
* **No tuning on test:** hyper-parameters are fixed in `config.yaml`; the primary model (`{pr}`) was named *before* looking at results. Ridge / gated variants are reported as ablations, not picked ex-post.
* Surprise z-scores use only *earlier* releases of the same series; news features use only articles with an exact timestamp strictly before the release.

{_t(fold_rows, ['Fold', 'Test start', 'Test end', 'Train rows', 'Test rows'])}

## 4. Results (out-of-sample, net of costs)
{perf}

* Primary Sharpe 95% block-bootstrap CI: **[{ci[0]:.2f}, {ci[1]:.2f}]**.
* Random-pick baseline (same slots, random permutation, {len(res['random_sharpes'])} draws): mean Sharpe **{res['random_sharpe_mean']:.2f}**; permutation p-value of the primary strategy: **{res['perm_p_value']:.3f}**.
* Sharpe/Sortino are annualised from **daily** returns (√252) on *all* business days in the OOS window (flat days included), Sortino uses downside deviation vs 0, Max DD is on the compounded equity curve.

![equity](figures/equity_curve.png)

### Per-fold stability
{fold_perf}

![folds](figures/walkforward_folds.png)

### Predictive quality
{pq}

Classifier (P(win)): AUC **{clf['auc']:.3f}**, base win rate {clf['base_win_rate']*100:.1f}%, win rate in top decile of P(win) {clf['win_rate_top_decile']*100:.1f}%.

![matrix](figures/pnl_matrix.png)

### Feature importance
![imp](figures/feature_importance.png)

Top drivers: {', '.join(f'`{i}`' for i in fi.head(6).index)}.

## 5. Visual matrix outputs
![sent](figures/sentiment_timeline.png)
![topic](figures/topic_sentiment_matrix.png)
![surprise](figures/surprise_matrix.png)
![scenario](figures/scenario_matrix.png)

## 6. Leakage & robustness checklist
- [x] Entry strictly after release timestamp; labels use only later bars.
- [x] Surprise normalisation is expanding/causal; scheduled-calendar features (time-to-next release) are public information.
- [x] Purged/embargoed walk-forward; training labels fully realised before test start.
- [x] Test suite (`pytest`) checks feature immutability under future-data perturbation, fold purge integrity and metric formulas.
- [x] Costs included; vol-scaled sizing uses trailing bars only.

## 7. Limitations (read before trusting any number)
* Small sample: {prov['n_slots']} releases in ~{cfg['project']['lookback_days']} days. Per-fold Sharpe is noisy; the permutation test and bootstrap CI matter more than the point estimate.
* 36 permutations per release invites selection effects; the walk-forward + fixed hyper-parameters limit but do not remove this.
* Lexicon sentiment is crude; Tavily results are search-ranked (not a complete news archive) and undated items are excluded from features.
* Simple cost model (fixed bps), no slippage/market impact, futures roll/gaps ignored, hourly bars only.
* Only USD macro releases; regime shifts (e.g. policy pivots) can break learned relationships.

## 8. Reproduce
```bash
pip install -e . && cp .env.example .env   # add APIFY_TOKEN, TAVILY_API_KEY
python -m macro live      # ingest → features → walk-forward → charts → reports
pytest -q
```
"""
    path.write_text(md, encoding="utf-8")
