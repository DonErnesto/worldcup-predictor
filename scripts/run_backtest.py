#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.backtest import run_backtest_from_file


def main() -> None:
    predictions, summary = run_backtest_from_file()
    print(f"wrote {len(predictions)} predictions to test_results/backtest_predictions.csv")
    print("wrote summary to test_results/backtest_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
