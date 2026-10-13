"""CLI:  python -m macro demo | live | ingest | build | train | report"""
from __future__ import annotations
import argparse
import logging
from .config import load_config
from . import pipeline as P


def main() -> None:
    ap = argparse.ArgumentParser(prog="macro")
    ap.add_argument("cmd", choices=["demo", "live", "ingest", "build", "train", "report"])
    ap.add_argument("--mode", choices=["demo", "live"], default="live", help="data mode for ingest/build/train/report")
    ap.add_argument("--refresh", action="store_true", help="ignore cached Apify runs")
    ap.add_argument("--llm", action="store_true", help="polish macro_outlook.md with Claude (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = P.cfg_for(a.mode)
    if a.cmd in ("demo", "live"):
        P.run_all(a.cmd, a.refresh, a.llm)
    elif a.cmd == "ingest":
        P.ingest(cfg, a.mode, a.refresh)
    elif a.cmd == "build":
        P.build(cfg)
    elif a.cmd == "train":
        P.train(cfg)
    elif a.cmd == "report":
        P.report(cfg, a.llm)


if __name__ == "__main__":
    main()
