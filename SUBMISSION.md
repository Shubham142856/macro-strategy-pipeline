# Submission Checklist & Email (to gilad@crowdwisdomtrading.com)

1. Verify real data pipeline run: `reports/evaluation_report.md` shows `Data mode: live`.
2. Verify all 8 figures in `reports/figures/` are generated and up-to-date.
3. Verify `data/raw/manifest.json` and cached `data/raw/apify/*.json` are committed (enables instant offline reproduction for the reviewer without consuming API credits).
4. Create a repository on GitHub (e.g. `macro-strategy-pipeline`), add the remote, and push:
   ```bash
   git remote add origin https://github.com/<your-username>/macro-strategy-pipeline.git
   git branch -M main
   git push -u origin main
   ```
5. Send the email using the template below.

---

### Submission Email Template

**To:** gilad@crowdwisdomtrading.com  
**Subject:** Quant Data Scientist Intern Assessment — Macro Strategy Predictor — Shubham Raikwar

Hi Gilad,

Please find my completed assessment submission for the Quantitative Data Scientist Intern position:

1. **Repository:**
   - URL: `https://github.com/Shubham142856/macro-strategy-pipeline`

2. **Authentication & Scraping Provenance:**
   - **Security Best Practice:** In adherence to quantitative engineering and security best practices, personal API keys and tokens are not shared or hardcoded. The pipeline authenticates strictly via environment variables in a local `.env` file (see `.env.example`).
   - **Apify Actor:** `xtracto/forexfactory-calendar` (Apify Store)
   - **Tavily Search API:** Used for real-time macroeconomic news ingestion.
   - **Zero-Token Reproduction (Immediate Testing):** All 10 live Apify runs (covering the 180-day trading window + 365-day surprise warmup, 830 real USD events) and news items are already cached under `data/raw/apify/` and fully cataloged in `data/raw/manifest.json`. You can clone and reproduce the entire pipeline end-to-end immediately without needing any API keys or consuming credits.
   - **Live Scraping with Custom Keys:** To trigger a fresh live scrape against the external APIs, you can provide your own `APIFY_TOKEN` and `TAVILY_API_KEY` in `.env` as documented in `README.md`.

3. **Evaluation Summary & Deliverables:**
   - **Full Evaluation Report:** [`reports/evaluation_report.md`](reports/evaluation_report.md)
   - **1-Page Macro Outlook:** [`reports/macro_outlook.md`](reports/macro_outlook.md)
   - **Visual Output Charts (8 figures):** Located in [`reports/figures/`](reports/figures/)
     - `sentiment_timeline.png`: News sentiment and policy tone mapped against key FOMC, CPI, and NFP releases.
     - `topic_sentiment_matrix.png`: Heatmap of macro topic sentiment across weekly horizons.
     - `surprise_matrix.png`: Surprise z-scores across inflation, labor, growth, and policy themes.
     - `scenario_matrix.png`: Model predicted P&L (bps) for upcoming catalysts across below/inline/above consensus scenarios.
     - `equity_curve.png`: Out-of-sample compounded equity curve net of transaction costs vs. baselines.
     - `walkforward_folds.png`: 5-fold expanding window validation performance and train/test boundaries.
     - `pnl_matrix.png`: Performance breakdown by asset and holding horizon (1h, 4h, 24h).
     - `feature_importance.png`: Feature importance across macro signals, sentiment, and time features.

4. **Model Performance Metrics (Out-of-Sample, Net of Costs):**
   - **Primary Model (HistGradientBoostingRegressor):**
     - **Sharpe Ratio:** 0.28 (annualized daily, flat days included)
     - **Sortino Ratio:** 0.43 (downside deviation vs 0)
     - **Max Drawdown:** -3.6%
     - **Annualized Return:** +1.5%
     - **Trades Traded:** 65
   - **Gated Strategy (GBM + P(win) ≥ 0.5 classifier gate):**
     - **Sharpe Ratio:** 0.72
     - **Sortino Ratio:** 1.21
     - **Max Drawdown:** -2.0%
     - **Win Rate:** 57%
     - **Profit Factor:** 1.17
   - **Linear Baseline (Ridge):** Sharpe -1.49, Sortino -1.98, Max DD -2.3%
   - **Random Selection Baseline (300 draws):** Mean Sharpe -0.36

5. **How to Reproduce:**
   ```bash
   git clone <REPO_URL>
   cd macro-strategy-pipeline
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   pytest -q                  # 13 tests pass
   python -m macro live       # Runs live evaluation using cached raw data / live APIs
   ```

Looking forward to your feedback and discussing the pipeline further.

Best regards,  
Shubham Raikwar  
shubhamraikwar2856@gmail.com
