#!/usr/bin/env python3
"""Download ERCOT load, hub prices, and Open-Meteo weather into data/processed."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ercot_forecast.download import build_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-download files that are already in data/raw")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frame = build_dataset(refresh=args.refresh)
    print(f"Cached {len(frame):,} hourly rows at data/processed/ercot_hourly.parquet")


if __name__ == "__main__":
    main()
