"""Macro calendar via Apify (default actor: xtracto/forexfactory-calendar).

Every run is cached to data/raw/apify/*.json and logged to data/raw/manifest.json
(actor id, run id, dataset id, window) so the pipeline is reproducible.
"""
from __future__ import annotations
import copy
import json
import logging
import os
from datetime import datetime, timezone
import pandas as pd
from ..config import data_path
from .calendar_parse import normalize_calendar

log = logging.getLogger(__name__)


def _fill(obj, mapping):
    if isinstance(obj, dict):
        return {k: _fill(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill(v, mapping) for v in obj]
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("{" + k + "}", v)
        return obj
    return obj


def _windows(start: pd.Timestamp, end: pd.Timestamp, step: int):
    cur = start.normalize()
    while cur <= end:
        nxt = min(cur + pd.Timedelta(days=step - 1), end.normalize())
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt + pd.Timedelta(days=1)


def _log_manifest(entry: dict) -> None:
    p = data_path("data", "raw", "manifest.json")
    m = json.loads(p.read_text()) if p.exists() else []
    m.append(entry)
    p.write_text(json.dumps(m, indent=2))


def fetch_calendar(cfg: dict, start: pd.Timestamp, end: pd.Timestamp, refresh: bool = False) -> pd.DataFrame:
    from apify_client import ApifyClient
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_TOKEN missing (set it in .env)")
    client = ApifyClient(token)
    a = cfg["apify"]
    items_all: list[dict] = []
    for frm, to in _windows(start, end, a.get("chunk_days", 60)):
        got = None
        for spec in a["actors"]:
            actor = spec["actor_id"]
            cache = data_path("data", "raw", "apify", f"{actor.replace('/', '__')}_{frm}_{to}.json")
            if cache.exists() and not refresh:
                got = json.loads(cache.read_text())
                log.info("cache hit %s %s..%s (%d rows)", actor, frm, to, len(got))
                break
            try:
                run_input = _fill(copy.deepcopy(spec["input"]), {"from": frm, "to": to})
                timeout_s = int(a.get("timeout_secs", 1800))
                try:
                    from datetime import timedelta
                    run = client.actor(actor).call(run_input=run_input, wait_duration=timedelta(seconds=timeout_s),
                                                   run_timeout=timedelta(seconds=timeout_s))
                except TypeError:
                    run = client.actor(actor).call(run_input=run_input, timeout_secs=timeout_s)
                run_id = getattr(run, "id", None) if hasattr(run, "id") else (run.get("id") if isinstance(run, dict) else None)
                ds_id = getattr(run, "default_dataset_id", None) if hasattr(run, "default_dataset_id") else (
                    (run.get("defaultDatasetId") or run.get("default_dataset_id") or run.get("datasetId")) if isinstance(run, dict) else None
                )
                got = list(client.dataset(ds_id).iterate_items())
                cache.write_text(json.dumps(got))
                _log_manifest({"actor_id": actor, "run_id": run_id, "dataset_id": ds_id,
                               "from": frm, "to": to, "n_items": len(got), "input": run_input,
                               "fetched_at": datetime.now(timezone.utc).isoformat()})
                log.info("apify %s %s..%s -> %d rows", actor, frm, to, len(got))
                break
            except Exception as ex:  # try next actor
                log.warning("actor %s failed for %s..%s: %s", actor, frm, to, ex)
        if got is None:
            raise RuntimeError(f"All Apify actors failed for window {frm}..{to}")
        items_all.extend(got)
    df = normalize_calendar(items_all, cfg["project"]["display_tz"])
    if df.empty:
        raise RuntimeError("Calendar came back empty. Check actor input/proxy (actuals need residential proxy).")
    return df
