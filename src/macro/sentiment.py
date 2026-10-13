"""Transparent, dependency-free macro-news scorer.

sentiment  in [-1, 1] : risk-on / growth-positive (+)  vs  risk-off / growth-negative (-)
policy_tone in [-1, 1]: hawkish (+) vs dovish (-)
"""
from __future__ import annotations
import re
import numpy as np

POS = {
    "beat": 1, "beats": 1, "stronger": 1, "strong": 0.8, "surge": 1, "surges": 1, "rally": 1, "rallies": 1,
    "rebound": 1, "resilient": 1, "expands": 1, "expansion": 0.8, "upbeat": 1, "optimism": 1, "soft landing": 1.5,
    "cooling inflation": 1, "disinflation": 1, "rate cut": 1, "rate cuts": 1, "easing": 0.7, "record high": 1,
    "improves": 0.8, "improved": 0.8, "gains": 0.7, "recovery": 0.8, "hiring": 0.5, "growth": 0.4, "jumps": 0.7,
}
NEG = {
    "recession": 1.5, "slump": 1, "contraction": 1, "contracts": 1, "weak": 1, "weaker": 1, "miss": 1, "misses": 1,
    "slowdown": 1, "layoffs": 1, "downturn": 1, "tariff": 0.7, "tariffs": 0.7, "crisis": 1.2, "selloff": 1,
    "sell-off": 1, "plunge": 1.2, "plunges": 1.2, "stagflation": 1.5, "fears": 0.7, "worries": 0.6, "slumps": 1,
    "deteriorates": 1, "falls": 0.5, "drops": 0.5, "shutdown": 0.8, "default": 0.8, "unrest": 0.8,
}
HAWK = {
    "hike": 1, "hikes": 1, "hawkish": 1.5, "higher for longer": 1.5, "tightening": 1, "hot inflation": 1.2,
    "sticky inflation": 1.2, "inflation accelerates": 1.2, "hotter": 1, "upside inflation": 1, "rate hike": 1.2,
    "inflation rises": 1, "inflation jumps": 1,
}
DOVE = {
    "dovish": 1.5, "rate cut": 1.2, "rate cuts": 1.2, "cuts rates": 1.2, "easing": 1, "disinflation": 1,
    "cooling inflation": 1.2, "inflation cools": 1.2, "cooler": 0.8, "pause": 0.8, "lower rates": 1,
    "inflation falls": 1,
}
TOPICS = {
    "inflation": ["inflation", "cpi", "pce", "ppi", "consumer prices", "price pressures"],
    "labor": ["jobs", "payroll", "payrolls", "unemployment", "labor market", "hiring", "layoffs", "wages", "jobless"],
    "growth": ["gdp", "growth", "manufacturing", "retail sales", "consumer spending", "pmi", "ism", "recession"],
    "policy": ["fed ", "fomc", "rate cut", "rate hike", "powell", "central bank", "interest rate", "rate decision"],
    "markets": ["stocks", "equities", "s&p", "nasdaq", "dollar", "treasury yields", "yields", "oil", "gold"],
}
_NEG_PREFIX = r"(?:not|no|never|without|fails? to|failed to|unlikely to)\s+(?:\w+\s+){0,2}"


def _score(text: str, lex: dict[str, float]) -> float:
    total = 0.0
    for phrase, w in lex.items():
        pat = r"\b" + re.escape(phrase) + r"\b"
        n = len(re.findall(pat, text))
        if n:
            neg = len(re.findall(_NEG_PREFIX + pat, text))
            total += w * (n - 2 * neg)
    return total


def score_text(text: str) -> dict:
    t = (text or "").lower()
    pos, neg = _score(t, POS), _score(t, {k: v for k, v in NEG.items()})
    hawk, dove = _score(t, HAWK), _score(t, DOVE)
    out = {
        "sentiment": float(np.clip((pos - neg) / (abs(pos) + abs(neg) + 1.0), -1, 1)),
        "policy_tone": float(np.clip((hawk - dove) / (abs(hawk) + abs(dove) + 1.0), -1, 1)),
        "recession_flag": int(bool(re.search(r"\brecession\b|\bstagflation\b|\bdownturn\b", t))),
    }
    for k, kws in TOPICS.items():
        out[f"topic_{k}"] = int(any(kw in t for kw in kws))
    return out
