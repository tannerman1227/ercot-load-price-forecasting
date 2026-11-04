#!/usr/bin/env python3
"""Fit the day-ahead models, score them, and write charts plus summary.json."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ercot_forecast.pipeline import run_analysis  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_analysis(save=True)
    headline = summary["headline"]
    load = headline["load_gbm_lagged"]
    naive = headline["load_naive_24"]
    print(
        "Load GBM MAE {gbm:,.0f} MW vs 24h naive {naive:,.0f} MW "
        "(skill {skill:.1%}, MAPE {mape:.2f}%).".format(
            gbm=load.get("mae", float("nan")),
            naive=naive.get("mae", float("nan")),
            skill=load.get("skill_vs_naive_24", float("nan")),
            mape=load.get("mape", float("nan")),
        )
    )
    print("Wrote data/processed/summary.json and reports/figures/")


if __name__ == "__main__":
    main()
