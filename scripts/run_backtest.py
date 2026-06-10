#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.backtest import PREDICTORS, run_backtest_from_file


def main() -> None:
    requested = sys.argv[1:] or list(PREDICTORS)
    for predictor_name in requested:
        predictions, summary = run_backtest_from_file(predictor_name=predictor_name)
        print(f"\n{predictor_name}")
        print(f"wrote {len(predictions)} predictions to test_results/{predictor_name}_backtest_predictions.csv")
        print(f"wrote summary to test_results/{predictor_name}_backtest_summary.csv")
        print(summary.to_string(index=False))

        if predictor_name == "phase_split_poisson":
            output_dir = REPO_ROOT / "test_results"
            (output_dir / "backtest_predictions.csv").write_text(
                (output_dir / f"{predictor_name}_backtest_predictions.csv").read_text()
            )
            (output_dir / "backtest_summary.csv").write_text(
                (output_dir / f"{predictor_name}_backtest_summary.csv").read_text()
            )


if __name__ == "__main__":
    main()
