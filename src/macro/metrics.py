from __future__ import annotations
import numpy as np
import pandas as pd

ANN = 252


def equity_curve(r: pd.Series) -> pd.Series:
    return (1 + r).cumprod()


def max_drawdown(r: pd.Series) -> float:
    eq = equity_curve(r)
    return float((eq / eq.cummax() - 1).min())


def perf_stats(r: pd.Series) -> dict:
    r = r.astype(float)
    sd = r.std(ddof=1)
    dd = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    eq = equity_curve(r)
    yrs = max(len(r) / ANN, 1e-9)
    ann_ret = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mdd = max_drawdown(r)
    return {
        "sharpe": float(r.mean() / sd * np.sqrt(ANN)) if sd > 0 else float("nan"),
        "sortino": float(r.mean() / dd * np.sqrt(ANN)) if dd > 0 else float("nan"),
        "max_drawdown": mdd, "ann_return": ann_ret, "ann_vol": float(sd * np.sqrt(ANN)),
        "calmar": float(ann_ret / abs(mdd)) if mdd < 0 else float("nan"),
        "total_return": float(eq.iloc[-1] - 1), "n_days": int(len(r)),
    }


def block_bootstrap_ci(r: pd.Series, fn, reps: int, block: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    x = r.values
    n = len(x)
    nb = int(np.ceil(n / block))
    vals = []
    for _ in range(reps):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        s = pd.Series(x[idx])
        vals.append(fn(s))
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def sharpe_fn(s: pd.Series) -> float:
    sd = s.std(ddof=1)
    return float(s.mean() / sd * np.sqrt(ANN)) if sd > 0 else np.nan


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"n_trades": 0}
    w, l = t.loc[t.pnl_bps > 0, "pnl_bps"], t.loc[t.pnl_bps <= 0, "pnl_bps"]
    return {"n_trades": int(len(t)), "win_rate": float((t.pnl_bps > 0).mean()), "avg_pnl_bps": float(t.pnl_bps.mean()),
            "profit_factor": float(w.sum() / abs(l.sum())) if l.sum() != 0 else float("nan")}
