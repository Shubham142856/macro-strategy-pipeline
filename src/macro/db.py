"""SQLite storage. All datasets are joined *in the database* (see features.news_features_sql)."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd
from .config import ROOT


def connect(cfg: dict) -> sqlite3.Connection:
    p = ROOT / cfg["project"]["db_path"]
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL")
    return con


def write_table(con, name: str, df: pd.DataFrame, index: list[str] | None = None) -> None:
    df.to_sql(name, con, if_exists="replace", index=False)
    for col in index or []:
        con.execute(f"CREATE INDEX IF NOT EXISTS ix_{name}_{col} ON {name}({col})")
    con.commit()


def read_table(con, name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {name}", con)


def set_meta(con, key: str, value: str) -> None:
    con.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (key, str(value)))
    con.commit()


def get_meta(con, key: str, default: str | None = None) -> str | None:
    try:
        r = con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return default
    return r[0] if r else default
