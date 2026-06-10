#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from worldcup_predictor.site_data import write_site_payload


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "site/data/predictions.json"
    payload = write_site_payload(output_path=output_path)
    print(f"wrote {len(payload['teams'])} teams")
    print(f"wrote {len(payload['predictions'])} predictions")
    print(f"wrote {len(payload['head_to_head'])} head-to-head pairs")
    print(f"output: {output_path}")


if __name__ == "__main__":
    main()
