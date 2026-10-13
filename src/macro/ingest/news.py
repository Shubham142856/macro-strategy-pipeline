"""News via Tavily (primary, per assignment) and optionally Exa.

Look-ahead protection: only items with an *exact* `published_date` are used as model features
(date_quality == 'exact'); undated items are kept for the written summary only.
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import time
import pandas as pd
import requests
from ..config import data_path
from ..sentiment import score_text

log = logging.getLogger(__name__)
TAVILY_URL = "https://api.tavily.com/search"
EXA_URL = "https://api.exa.ai/search"


def _cached_post(url: str, headers: dict, payload: dict, tag: str, retries: int = 3) -> dict:
    key = hashlib.md5((url + json.dumps(payload, sort_keys=True)).encode()).hexdigest()
    cache = data_path("data", "raw", tag, f"{key}.json")
    if cache.exists():
        return json.loads(cache.read_text())
    for i in range(retries):
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** (i + 1))
            continue
        if r.status_code >= 400:
            log.warning("%s %s -> %s %s", tag, payload.get("query"), r.status_code, r.text[:200])
            return {}
        out = r.json()
        cache.write_text(json.dumps(out))
        return out
    return {}


def _weekly(start: pd.Timestamp, end: pd.Timestamp, step: int):
    cur = start.normalize()
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=step), end.normalize() + pd.Timedelta(days=1))
        yield cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")
        cur = nxt


def fetch_tavily(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("TAVILY_API_KEY missing")
    t = cfg["tavily"]
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    out = []
    # 1) historical weekly windows (start/end dates)  2) the latest week via topic=news
    for q in t["queries"]:
        for s, e in _weekly(start, end, t.get("window_days", 7)):
            body = {"query": q, "topic": "news", "search_depth": t.get("search_depth", "basic"),
                    "max_results": t.get("max_results", 20), "start_date": s, "end_date": e}
            res = _cached_post(TAVILY_URL, hdr, body, "tavily")
            for r in res.get("results", []):
                out.append({**r, "query": q, "source_api": "tavily"})
        body = {"query": q, "topic": "news", "time_range": "week", "search_depth": t.get("search_depth", "basic"),
                "max_results": t.get("max_results", 20)}
        res = _cached_post(TAVILY_URL, hdr, body, "tavily_latest")
        for r in res.get("results", []):
            out.append({**r, "query": q, "source_api": "tavily"})
    return out


def fetch_exa(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    key = os.environ.get("EXA_API_KEY")
    if not key or not cfg.get("exa", {}).get("enabled"):
        return []
    hdr = {"x-api-key": key, "Content-Type": "application/json"}
    out = []
    for q in cfg["tavily"]["queries"]:
        for s, e in _weekly(start, end, cfg["tavily"].get("window_days", 7)):
            body = {"query": q, "category": "news", "numResults": cfg["exa"].get("num_results", 25),
                    "startPublishedDate": f"{s}T00:00:00.000Z", "endPublishedDate": f"{e}T00:00:00.000Z",
                    "contents": {"text": {"maxCharacters": 1200}}}
            res = _cached_post(EXA_URL, hdr, body, "exa")
            for r in res.get("results", []):
                out.append({"title": r.get("title"), "url": r.get("url"), "content": r.get("text"),
                            "published_date": r.get("publishedDate"), "query": q, "source_api": "exa"})
    return out


def normalize_news(items: list[dict]) -> pd.DataFrame:
    rows = []
    for r in items:
        title, content, url = r.get("title") or "", r.get("content") or "", r.get("url") or ""
        pub = pd.to_datetime(r.get("published_date"), utc=True, errors="coerce")
        sc = score_text(f"{title}. {content}")
        rows.append({
            "news_id": hashlib.md5((url or title).encode()).hexdigest()[:16],
            "published_utc": int(pub.timestamp()) if pd.notna(pub) else None,
            "date_quality": "exact" if pd.notna(pub) else "missing",
            "title": title[:300], "content": content[:1500], "url": url, "query": r.get("query"),
            "source_api": r.get("source_api"), **sc,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("published_utc").drop_duplicates("news_id").reset_index(drop=True)
    return df


def fetch_rss_news(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    """Real-world financial RSS news fallback when TAVILY_API_KEY is not supplied."""
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET
    out = []
    t = cfg.get("tavily", {})
    queries = t.get("queries", ["US inflation FOMC rate decision economy"])
    for q in queries:
        query_enc = urllib.parse.quote_plus(f"{q} when:180d")
        url = f"https://news.google.com/rss/search?q={query_enc}&hl=en-US&gl=US&ceid=US:en"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                tree = ET.fromstring(resp.read())
                for item in tree.findall(".//item"):
                    title = item.find("title").text if item.find("title") is not None else ""
                    pub_date = item.find("pubDate").text if item.find("pubDate") is not None else None
                    link = item.find("link").text if item.find("link") is not None else ""
                    desc = item.find("description").text if item.find("description") is not None else ""
                    out.append({
                        "title": title, "content": desc or title, "url": link,
                        "published_date": pub_date, "query": q, "source_api": "rss_macro"
                    })
        except Exception as e:
            log.warning("RSS news fetch error for '%s': %s", q, e)
    return out


def fetch_news(cfg: dict, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    items = []
    if os.environ.get("TAVILY_API_KEY"):
        items += fetch_tavily(cfg, start, end)
    elif not os.environ.get("EXA_API_KEY"):
        log.info("TAVILY_API_KEY not found in environment; using live financial RSS macroeconomic news.")
        items += fetch_rss_news(cfg, start, end)
    if os.environ.get("EXA_API_KEY"):
        items += fetch_exa(cfg, start, end)
    if not items and not os.environ.get("TAVILY_API_KEY"):
        items += fetch_rss_news(cfg, start, end)
    df = normalize_news(items)
    if df.empty:
        raise RuntimeError("No news returned (check TAVILY_API_KEY / network connection).")
    return df
