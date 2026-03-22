#!/usr/bin/env python
"""Prepare the paper-style 110-class single-label dataset for charge prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.constants import DEFAULT_RANDOM_SEED, PAPER_TOP_LEVEL_CATEGORIES
from charge_prediction.data_utils import (
    build_single_label_paper_dataset,
    dataframe_to_records,
    dataset_basic_stats,
    export_processed_analysis_tables,
    parse_single_label_cail_split,
    split_sample_sizes,
    stratified_sample_df,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 110-class single-label paper dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("../2018数据集/2018数据集"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_110_paper"))
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--top-k-labels", type=int, default=110)
    parser.add_argument("--task-mode", type=str, default="single_label_110", choices=["single_label_110"])
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--analysis-dir", type=Path, default=None)
    parser.add_argument("--skip-export-tables", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = parse_single_label_cail_split(args.data_dir / "data_train.json", PAPER_TOP_LEVEL_CATEGORIES)
    valid_df = parse_single_label_cail_split(args.data_dir / "data_valid.json", PAPER_TOP_LEVEL_CATEGORIES)
    test_df = parse_single_label_cail_split(args.data_dir / "data_test.json", PAPER_TOP_LEVEL_CATEGORIES)

    raw_sizes = {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }

    train_df, valid_df, test_df, accusation_to_category, top_labels = build_single_label_paper_dataset(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        top_k_labels=args.top_k_labels,
    )

    filtered_sizes = {
        "train": len(train_df),
        "valid": len(valid_df),
        "test": len(test_df),
    }

    train_target, valid_target, test_target = split_sample_sizes(
        len(train_df),
        len(valid_df),
        len(test_df),
        args.target_size,
    )

    train_sample = stratified_sample_df(train_df, sample_size=train_target, seed=args.seed)
    valid_sample = stratified_sample_df(valid_df, sample_size=valid_target, seed=args.seed)
    test_sample = stratified_sample_df(test_df, sample_size=test_target, seed=args.seed)

    write_jsonl(dataframe_to_records(train_sample), args.output_dir / "train_50k.jsonl")
    write_jsonl(dataframe_to_records(valid_sample), args.output_dir / "valid_50k.jsonl")
    write_jsonl(dataframe_to_records(test_sample), args.output_dir / "test_50k.jsonl")

    fine_labels = sorted(train_sample["fine_label"].unique().tolist())
    coarse_labels = sorted(train_sample["coarse_label"].unique().tolist())

    (args.output_dir / "label2id.json").write_text(
        json.dumps({label: index for index, label in enumerate(fine_labels)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "coarse_label2id.json").write_text(
        json.dumps({label: index for index, label in enumerate(coarse_labels)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "accusation_to_category.json").write_text(
        json.dumps(accusation_to_category, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "top_k_labels.json").write_text(
        json.dumps(top_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "task_mode": args.task_mode,
        "raw_sizes": raw_sizes,
        "filtered_sizes": filtered_sizes,
        "train": dataset_basic_stats(train_sample),
        "valid": dataset_basic_stats(valid_sample),
        "test": dataset_basic_stats(test_sample),
        "target_total": args.target_size,
        "actual_total": len(train_sample) + len(valid_sample) + len(test_sample),
        "train_target": train_target,
        "valid_target": valid_target,
        "test_target": test_target,
        "top_k_labels": args.top_k_labels,
        "paper_top_level_categories": PAPER_TOP_LEVEL_CATEGORIES,
    }

    if not args.skip_export_tables:
        analysis_dir = args.analysis_dir if args.analysis_dir is not None else args.output_dir / "analysis"
        stats["analysis_tables"] = export_processed_analysis_tables(
            train_df=train_sample,
            valid_df=valid_sample,
            test_df=test_sample,
            output_dir=analysis_dir,
        )

    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Done] paper-style dataset prepared")
    print(f"  train: {len(train_sample)}")
    print(f"  valid: {len(valid_sample)}")
    print(f"  test : {len(test_sample)}")
    print(f"  fine labels  : {len(fine_labels)}")
    print(f"  coarse labels: {len(coarse_labels)}")
    print(f"  categories   : {', '.join(coarse_labels)}")


if __name__ == "__main__":
    main()
