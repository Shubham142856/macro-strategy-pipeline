from __future__ import annotations
from pathlib import Path
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    load_dotenv(ROOT / ".env")
    p = Path(path) if path else ROOT / "config.yaml"
    cfg = yaml.safe_load(p.read_text())
    cfg["_root"] = str(ROOT)
    return cfg


def data_path(*parts: str) -> Path:
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def sym_short(sym: str) -> str:
    return sym.replace("=F", "").replace("=X", "")


def epoch_s(idx):
    """tz-aware DatetimeIndex -> int64 epoch seconds (robust to pandas 2/3 datetime resolutions)."""
    import numpy as np
    import pandas as pd
    return np.asarray((idx - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1), dtype="int64")
