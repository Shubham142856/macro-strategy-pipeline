"""Visual matrix outputs (PNG). All figures are produced from the database tables, not from in-memory state."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .metrics import equity_curve

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})
KEY_COL = {"is_cpi": "#d62728", "is_fomc": "#1f77b4", "is_nfp": "#2ca02c"}


def _dt(ts):
    return pd.to_datetime(ts, unit="s", utc=True).dt.tz_localize(None)


def fig_sentiment_timeline(news: pd.DataFrame, slots: pd.DataFrame, path):
    n = news[news.date_quality == "exact"].copy()
    n["day"] = _dt(n.published_utc).dt.floor("D")
    d = n.groupby("day").agg(sent=("sentiment", "mean"), tone=("policy_tone", "mean"), cnt=("news_id", "count"))
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax[0].bar(d.index, d.sent, color=np.where(d.sent.fillna(0) >= 0, "#9ecae1", "#fcae91"), width=0.9)
    ax[0].plot(d.index, d.sent.ewm(span=7, min_periods=3).mean(), color="k", lw=1.8, label="EMA(7d)")
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_ylabel("Macro sentiment\n(risk-off ← → risk-on)")
    ax[0].set_title("Macro news sentiment over time (Tavily/Exa headlines) with key release markers")
    ax[0].legend(loc="upper left")
    ax[1].plot(d.index, d.tone.ewm(span=7, min_periods=3).mean(), color="#9467bd", lw=1.8)
    ax[1].fill_between(d.index, 0, d.tone.ewm(span=7, min_periods=3).mean(), alpha=0.2, color="#9467bd")
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_ylabel("Policy tone\n(dovish ← → hawkish)")
    ax[2].bar(d.index, d.cnt, color="#bbb")
    ax[2].set_ylabel("# articles")
    s = slots.copy()
    s["t"] = _dt(s.slot_ts)
    for flag, col in KEY_COL.items():
        for t in s.loc[s[flag] == 1, "t"]:
            for a in ax[:2]:
                a.axvline(t, color=col, alpha=0.35, lw=1)
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([0], [0], color="k", lw=1.8, label="EMA(7d)")] +
                 [Line2D([0], [0], color=c, alpha=0.6, label=l) for l, c in [("CPI", "#d62728"), ("FOMC", "#1f77b4"), ("Jobs/NFP", "#2ca02c")]],
                 loc="upper left", ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_topic_matrix(news: pd.DataFrame, path):
    n = news[news.date_quality == "exact"].copy()
    n["week"] = _dt(n.published_utc).dt.to_period("W").dt.start_time
    topics = ["inflation", "labor", "growth", "policy", "markets"]
    M = pd.DataFrame({t: n[n[f"topic_{t}"] == 1].groupby("week")["sentiment"].mean() for t in topics}).T
    P = pd.DataFrame({t: n[n[f"topic_{t}"] == 1].groupby("week")["policy_tone"].mean() for t in topics}).T
    fig, ax = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    for a, D, ttl, cm in [(ax[0], M, "Sentiment by macro topic × week (green = positive / risk-on)", "RdYlGn"),
                          (ax[1], P, "Policy tone by topic × week (red = hawkish)", "coolwarm")]:
        im = a.imshow(D.values.astype(float), aspect="auto", cmap=cm, vmin=-0.6, vmax=0.6)
        a.set_yticks(range(len(D))); a.set_yticklabels(D.index)
        step = max(1, len(D.columns) // 12)
        a.set_xticks(range(0, len(D.columns), step)); a.set_xticklabels([c.strftime("%b %d") for c in D.columns[::step]], rotation=0)
        a.set_title(ttl); a.grid(False)
        fig.colorbar(im, ax=a, fraction=0.02)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_surprise_matrix(ev: pd.DataFrame, path):
    e = ev[(ev.released == 1) & ev.signed_score.notna()].copy()
    e["week"] = _dt(e.ts_utc).dt.to_period("W").dt.start_time
    D = e.pivot_table(index="category", columns="week", values="signed_score", aggfunc="mean")
    D = D.reindex([c for c in ["inflation", "labor", "growth", "policy"] if c in D.index])
    fig, ax = plt.subplots(figsize=(11, 2.9))
    im = ax.imshow(D.values.astype(float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_yticks(range(len(D))); ax.set_yticklabels(D.index); ax.grid(False)
    step = max(1, len(D.columns) // 12)
    ax.set_xticks(range(0, len(D.columns), step)); ax.set_xticklabels([c.strftime("%b %d") for c in D.columns[::step]])
    ax.set_title("Data-surprise matrix: weekly mean surprise z-score (red = hotter/stronger than consensus, blue = cooler/weaker)")
    fig.colorbar(im, ax=ax, fraction=0.02); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_scenario(scen: pd.DataFrame, path):
    if scen is None or scen.empty:
        return
    h = scen.horizon.iloc[0]
    D = scen.pivot_table(index=["asset", "direction"], columns="scenario", values="pred_gbm")
    order = ["Below consensus", "In line", "Above consensus"]
    D = D[[c for c in order if c in D.columns]]
    labels = [f"{a} {'LONG' if d > 0 else 'SHORT'}" for a, d in D.index]
    fig, ax = plt.subplots(figsize=(6.2, 6.6))
    v = np.nanmax(np.abs(D.values)) or 1
    im = ax.imshow(D.values, aspect="auto", cmap="RdYlGn", vmin=-v, vmax=v)
    ax.set_yticks(range(len(D))); ax.set_yticklabels(labels); ax.set_xticks(range(D.shape[1])); ax.set_xticklabels(D.columns)
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            ax.text(j, i, f"{D.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    ax.grid(False)
    ax.set_title(f"Predicted net P&L (bps of capital, {h}h hold)\nnext key release: {scen.headline.iloc[0]}")
    fig.colorbar(im, ax=ax, fraction=0.04); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_equity(res: dict, path):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    lab = {"gbm": "GBM (primary)", "ridge": "Ridge", "gbm_gated": "GBM + P(win) gate", "naive_long_ES_4h": "Naive long ES 4h",
           "buy_hold_ES": "Buy & hold ES"}
    for k, l in lab.items():
        r = res["daily"].get(k)
        if r is None or len(r) < 2:
            continue
        eq = equity_curve(r)
        ax[0].plot(eq.index, (eq - 1) * 100, label=l, lw=2.4 if k == res["primary"] else 1.2, alpha=1 if k == res["primary"] else 0.8)
    ax[0].set_ylabel("Cumulative return (%)"); ax[0].legend(ncol=2, fontsize=8)
    ax[0].set_title("Out-of-sample equity curves (walk-forward, net of costs)")
    r = res["daily"][res["primary"]]
    eq = equity_curve(r)
    ax[1].fill_between(eq.index, (eq / eq.cummax() - 1) * 100, 0, color="#d62728", alpha=0.4)
    ax[1].set_ylabel("Drawdown (%)"); fig = ax[0].figure
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_folds(res: dict, fold_info: pd.DataFrame, path):
    f = res["folds"]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].bar(f.fold, f.sharpe, color=np.where(f.sharpe >= 0, "#2ca02c", "#d62728")); ax[0].set_title("Sharpe by fold"); ax[0].set_xlabel("fold")
    ax[1].bar(f.fold, f.rank_ic, color=np.where(f.rank_ic >= 0, "#1f77b4", "#d62728")); ax[1].set_title("Cross-sectional rank IC by fold"); ax[1].set_xlabel("fold")
    ax[2].hist(res["random_sharpes"], bins=25, color="#bbb"); ax[2].axvline(res["stats"][res["primary"]]["sharpe"], color="k", lw=2)
    ax[2].set_title(f"Random-pick Sharpe vs model (p={res['perm_p_value']:.3f})"); ax[2].set_xlabel("Sharpe")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_pnl_matrix(oos: pd.DataFrame, path):
    sel = oos[oos.pred_gbm > 0]
    A = sel.pivot_table(index="asset", columns="horizon", values="pnl_bps", aggfunc="mean")
    B = oos.pivot_table(index="asset", columns="horizon", values="pnl_bps", aggfunc="mean")
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for a, D, t in [(ax[0], B, "All permutations (unconditional)"), (ax[1], A.reindex_like(B), "Model-approved permutations (pred > 0)")]:
        v = np.nanmax(np.abs(np.r_[B.values.ravel(), A.values.ravel()])) or 1
        im = a.imshow(D.values.astype(float), cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
        a.set_xticks(range(D.shape[1])); a.set_xticklabels([f"{h}h" for h in D.columns]); a.set_yticks(range(len(D))); a.set_yticklabels(D.index)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                if np.isfinite(D.values[i, j]):
                    a.text(j, i, f"{D.values[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        a.set_title(t, fontsize=9); a.grid(False)
    fig.suptitle("Realised OOS mean net P&L (bps of capital) by asset × holding period", fontsize=10)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_importance(fi: pd.Series, path, top=15):
    s = fi.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(s.index, s.values, color="#1f77b4"); ax.set_title("Permutation importance (last fold, ΔMSE)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
