#!/usr/bin/env python
"""Export processed 50k jsonl files into CSV tables for analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.data_utils import export_processed_analysis_tables, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export processed jsonl files to CSV analysis tables")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/analysis"))
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def main() -> None:
    args = parse_args()
    train_df = load_split(args.data_dir / "train_50k.jsonl")
    valid_df = load_split(args.data_dir / "valid_50k.jsonl")
    test_df = load_split(args.data_dir / "test_50k.jsonl")

    table_paths = export_processed_analysis_tables(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        output_dir=args.output_dir,
    )

    print("[Done] analysis tables exported")
    print(json.dumps(table_paths, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
