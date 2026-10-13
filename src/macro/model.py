"""Feature assembly + purged expanding-window walk-forward validation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .config import sym_short

SURP = ["s_inflation", "s_labor", "s_growth", "s_policy"]
BASE = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "dom_frac", "n_events", "n_high", "hrs_since_prev",
        "hrs_to_next", "days_since_fomc", "days_to_fomc"]
FLAGS = ["is_cpi", "is_fomc", "is_nfp", "is_pce", "is_gdp", "is_pmi", "is_retail", "is_claims"]
NEWS = ["sent_24h", "sent_72h", "sent_delta", "news_n_24h", "tone_24h", "recession_24h"]
STATE = ["pre_ret_4h", "pre_ret_24h", "pre_vol_bps"]
INTER_SRC = SURP + ["sent_24h", "sent_delta", "tone_24h", "pre_ret_4h", "pre_ret_24h"]


def assemble(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Row-level design matrix. Every column is known at entry time."""
    assets = [sym_short(s) for s in cfg["markets"]["assets"]]
    X = df[BASE + FLAGS + SURP + NEWS + STATE + ["direction", "horizon"]].copy().astype(float)
    for a in assets:
        X[f"a_{a}"] = (df["asset"] == a).astype(float)
    for c in INTER_SRC:
        X[f"dir_x_{c}"] = df["direction"] * df[c]
    for a in assets:
        for c in SURP:
            X[f"{a}_dir_{c}"] = X[f"a_{a}"] * df["direction"] * df[c]
    return X.fillna(0.0)


def make_models(cfg: dict, seed: int):
    g, m = cfg["model"]["gbm"], cfg["model"]
    kw = dict(max_depth=g["max_depth"], learning_rate=g["learning_rate"], max_iter=g["max_iter"],
              min_samples_leaf=g["min_samples_leaf"], l2_regularization=g["l2_regularization"], random_state=seed)
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=m["ridge_alpha"])),
        "gbm": HistGradientBoostingRegressor(loss="squared_error", **kw),
        "clf": HistGradientBoostingClassifier(**kw),
    }


def _winsor(y: np.ndarray, pct: float) -> np.ndarray:
    lo, hi = np.percentile(y, [pct, 100 - pct])
    return np.clip(y, lo, hi)


def fit_models(X: pd.DataFrame, y: pd.Series, win: pd.Series, cfg: dict):
    ms = make_models(cfg, cfg["project"]["seed"])
    yw = _winsor(y.values, cfg["model"]["winsor_pct"])          # train-only winsorisation
    ms["ridge"].fit(X, yw)
    ms["gbm"].fit(X, yw)
    ms["clf"].fit(X, win.values)
    return ms


def predict(ms, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"pred_ridge": ms["ridge"].predict(X), "pred_gbm": ms["gbm"].predict(X),
                         "p_win": ms["clf"].predict_proba(X)[:, 1]}, index=X.index)


def walk_forward(rows: pd.DataFrame, cfg: dict):
    """Expanding window over release slots. Training rows must have EXITED (label fully realised) at least
    `embargo_hours` before the first test entry -> no label overlap / leakage across the train/test boundary."""
    v = cfg["validation"]
    rows = rows.sort_values(["slot_ts", "asset", "direction", "horizon"]).reset_index(drop=True)
    X = assemble(rows, cfg)
    slots = np.sort(rows["slot_ts"].unique())
    n = len(slots)
    edges = np.linspace(int(n * v["initial_train_frac"]), n, v["n_folds"] + 1).astype(int)
    emb = v["embargo_hours"] * 3600
    parts, info = [], []
    last = None
    for k in range(v["n_folds"]):
        test_slots = slots[edges[k]:edges[k + 1]]
        t0 = test_slots[0]
        tr = (rows["exit_ts"] <= t0 - emb).values
        te = rows["slot_ts"].isin(test_slots).values
        ms = fit_models(X[tr], rows.loc[tr, "pnl_bps"], rows.loc[tr, "win"], cfg)
        pr = predict(ms, X[te])
        part = pd.concat([rows.loc[te, ["slot_ts", "asset", "direction", "horizon", "entry_ts", "exit_ts", "gross_bps",
                                        "pnl_bps", "win"]], pr], axis=1)
        part["fold"] = k
        parts.append(part)
        info.append({"fold": k, "train_rows": int(tr.sum()), "test_rows": int(te.sum()), "n_test_slots": len(test_slots),
                     "train_end_exit_utc": pd.to_datetime(rows.loc[tr, "exit_ts"].max(), unit="s", utc=True),
                     "test_start_utc": pd.to_datetime(test_slots[0], unit="s", utc=True),
                     "test_end_utc": pd.to_datetime(test_slots[-1], unit="s", utc=True)})
        last = (ms, X[te], rows.loc[te, "pnl_bps"])
    oos = pd.concat(parts, ignore_index=True)
    # permutation importance on the last fold (diagnostic only)
    ms, Xte, yte = last
    imp = permutation_importance(ms["gbm"], Xte, yte, n_repeats=3, random_state=cfg["project"]["seed"],
                                 scoring="neg_mean_squared_error")
    fi = pd.Series(imp.importances_mean, index=Xte.columns).sort_values(ascending=False)
    return oos, pd.DataFrame(info), fi


def fit_final(rows: pd.DataFrame, cfg: dict):
    X = assemble(rows, cfg)
    return fit_models(X, rows["pnl_bps"], rows["win"], cfg)
