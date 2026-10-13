"""Turn out-of-sample predictions into trades, daily returns, metrics, baselines and a permutation test."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, roc_auc_score
from . import metrics as M


def select_trades(oos: pd.DataFrame, pred_col: str, thr: float, gate: pd.Series | None = None) -> pd.DataFrame:
    d = oos if gate is None else oos[gate]
    if d.empty:
        return d
    idx = d.groupby("slot_ts")[pred_col].idxmax()
    t = d.loc[idx]
    return t[t[pred_col] > thr]


def daily_returns(trades: pd.DataFrame, start_ts: int, end_ts: int) -> pd.Series:
    """P&L is booked on the (UTC) day the position exits; weekend exits roll to Monday."""
    d = pd.to_datetime(trades["exit_ts"], unit="s", utc=True).dt.tz_localize(None).dt.floor("D")
    dow = d.dt.dayofweek
    d = d + pd.to_timedelta(np.where(dow == 5, 2, np.where(dow == 6, 1, 0)), unit="D")
    s = (trades["pnl_bps"].values / 1e4)
    s = pd.Series(s, index=d.values).groupby(level=0).sum()
    idx = pd.bdate_range(pd.to_datetime(start_ts, unit="s").floor("D"), pd.to_datetime(end_ts, unit="s").floor("D") + pd.Timedelta(days=3))
    return s.reindex(idx, fill_value=0.0)


def buy_hold(prices: pd.DataFrame, sym: str, start_ts: int, end_ts: int) -> pd.Series:
    p = prices[(prices.symbol == sym) & (prices.ts_utc >= start_ts) & (prices.ts_utc <= end_ts)].sort_values("ts_utc")
    d = pd.to_datetime(p.ts_utc, unit="s").dt.floor("D")
    daily = p.groupby(d.values)["close"].last()
    return daily.pct_change().dropna()


def evaluate(oos: pd.DataFrame, prices: pd.DataFrame, cfg: dict) -> dict:
    m, v = cfg["model"], cfg["validation"]
    t0, t1 = int(oos["entry_ts"].min()), int(oos["exit_ts"].max())
    res: dict = {"window": (t0, t1)}

    # ---- predictive quality (pooled + per-release cross-sectional rank IC) -------------------
    q = {}
    for name, col in [("ridge", "pred_ridge"), ("gbm", "pred_gbm")]:
        ics = []
        for _, g in oos.groupby("slot_ts"):
            if g[col].nunique() > 1 and g["pnl_bps"].nunique() > 1:
                ics.append(spearmanr(g[col], g["pnl_bps"])[0])
        ic = pd.Series(ics)
        q[name] = {"pooled_corr": float(np.corrcoef(oos[col], oos["pnl_bps"])[0, 1]), "r2": float(r2_score(oos["pnl_bps"], oos[col])),
                   "rank_ic_mean": float(ic.mean()), "rank_ic_t": float(ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))) if len(ic) > 2 else np.nan,
                   "ic_hit_rate": float((ic > 0).mean())}
    q["classifier"] = {"auc": float(roc_auc_score(oos["win"], oos["p_win"])), "base_win_rate": float(oos["win"].mean()),
                       "win_rate_top_decile": float(oos.loc[oos.p_win >= oos.p_win.quantile(0.9), "win"].mean())}
    res["predictive"] = q

    # ---- strategies --------------------------------------------------------------------------
    thr = m["trade_threshold_bps"]
    strat = {
        "gbm": select_trades(oos, "pred_gbm", thr),
        "ridge": select_trades(oos, "pred_ridge", thr),
        "gbm_gated": select_trades(oos, "pred_gbm", thr, gate=(oos.p_win >= m["min_p_win"])),
    }
    # naive baselines
    nv = oos[(oos.asset == "ES") & (oos.direction == 1) & (oos.horizon == 4)]
    strat["naive_long_ES_4h"] = nv
    rng = np.random.default_rng(cfg["project"]["seed"])
    groups = [g for _, g in oos.groupby("slot_ts")]
    rand_sharpes = []
    for _ in range(v["n_random_draws"]):
        picks = pd.DataFrame([g.iloc[rng.integers(len(g))] for g in groups])
        rand_sharpes.append(M.sharpe_fn(daily_returns(picks, t0, t1)))
    res["random_sharpes"] = np.array(rand_sharpes)

    daily, stats = {}, {}
    for k, t in strat.items():
        r = daily_returns(t, t0, t1) if len(t) else pd.Series(0.0, index=pd.bdate_range(pd.to_datetime(t0, unit="s").floor("D"), periods=10))
        daily[k] = r
        stats[k] = {**M.perf_stats(r), **M.trade_stats(t)}
    bh = buy_hold(prices, "ES=F", t0, t1)
    daily["buy_hold_ES"] = bh
    stats["buy_hold_ES"] = {**M.perf_stats(bh), "n_trades": 1}
    primary = m["primary"]
    r = daily[primary]
    lo, hi = M.block_bootstrap_ci(r, M.sharpe_fn, v["bootstrap_reps"], v["bootstrap_block"], cfg["project"]["seed"])
    stats[primary]["sharpe_ci95"] = (lo, hi)
    p_val = float((np.nansum(res["random_sharpes"] >= stats[primary]["sharpe"]) + 1) / (len(res["random_sharpes"]) + 1))
    res.update({"stats": stats, "daily": daily, "trades": strat, "primary": primary, "perm_p_value": p_val,
                "random_sharpe_mean": float(np.nanmean(res["random_sharpes"]))})

    # per-fold primary stats
    folds = []
    t = strat[primary]
    for k, g in oos.groupby("fold"):
        tk = t[t.fold == k]
        rk = daily_returns(tk, int(g.entry_ts.min()), int(g.exit_ts.max())) if len(tk) else pd.Series([0.0])
        ics = [spearmanr(gg["pred_gbm"], gg["pnl_bps"])[0] for _, gg in g.groupby("slot_ts") if gg["pred_gbm"].nunique() > 1 and gg["pnl_bps"].nunique() > 1]
        folds.append({"fold": k, "n_trades": len(tk), "sharpe": M.sharpe_fn(rk), "total_return_pct": float(M.equity_curve(rk).iloc[-1] - 1) * 100,
                      "rank_ic": float(np.mean(ics)) if ics else np.nan})
    res["folds"] = pd.DataFrame(folds)
    return res
