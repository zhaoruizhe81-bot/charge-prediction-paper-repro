#!/usr/bin/env python
"""Prepare the paper-style 110-class single-label dataset for charge prediction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.constants import DEFAULT_RANDOM_SEED, PAPER_TOP_LEVEL_CATEGORIES
from charge_prediction.data_utils import (
    DatasetBundle,
    build_accusation_to_category_from_df,
    dataframe_to_records,
    dataset_basic_stats,
    export_processed_analysis_tables,
    parse_single_label_cail_split,
    rebuild_stratified_splits,
    select_top_fine_labels,
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
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="rebuild_stable",
        choices=["rebuild_stable", "official_filtered"],
    )
    parser.add_argument("--min-label-support", type=int, default=3)
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

    pooled_df = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)
    top_labels, eligible_counts = select_top_fine_labels(
        pooled_df,
        top_k_labels=args.top_k_labels,
        min_label_support=args.min_label_support,
    )
    if not top_labels:
        raise ValueError(
            "No labels matched the requested support threshold. "
            "Try lowering --min-label-support or --top-k-labels."
        )

    top_label_set = set(top_labels)
    official_bundle = DatasetBundle(
        train=train_df[train_df["fine_label"].isin(top_label_set)].reset_index(drop=True),
        valid=valid_df[valid_df["fine_label"].isin(top_label_set)].reset_index(drop=True),
        test=test_df[test_df["fine_label"].isin(top_label_set)].reset_index(drop=True),
    )

    source_filtered_sizes = {
        "train": len(official_bundle.train),
        "valid": len(official_bundle.valid),
        "test": len(official_bundle.test),
    }

    if args.split_strategy == "official_filtered":
        filtered_bundle = official_bundle
    else:
        selected_counts = pooled_df[pooled_df["fine_label"].isin(top_label_set)]["fine_label"].value_counts()
        if int(selected_counts.min()) < 3:
            raise ValueError(
                "Stable rebuild requires every selected label to have at least 3 samples. "
                "Increase --min-label-support or reduce --top-k-labels."
            )

        filtered_bundle = rebuild_stratified_splits(
            pooled_df[pooled_df["fine_label"].isin(top_label_set)].reset_index(drop=True),
            train_ratio=source_filtered_sizes["train"],
            valid_ratio=source_filtered_sizes["valid"],
            test_ratio=source_filtered_sizes["test"],
            seed=args.seed,
            min_count_per_split=1,
        )

    accusation_to_category = build_accusation_to_category_from_df(
        pd.concat([filtered_bundle.train, filtered_bundle.valid, filtered_bundle.test], axis=0, ignore_index=True)
    )

    filtered_sizes = {
        "train": len(filtered_bundle.train),
        "valid": len(filtered_bundle.valid),
        "test": len(filtered_bundle.test),
    }

    train_target, valid_target, test_target = split_sample_sizes(
        len(filtered_bundle.train),
        len(filtered_bundle.valid),
        len(filtered_bundle.test),
        args.target_size,
    )

    train_sample = stratified_sample_df(filtered_bundle.train, sample_size=train_target, seed=args.seed)
    valid_sample = stratified_sample_df(filtered_bundle.valid, sample_size=valid_target, seed=args.seed)
    test_sample = stratified_sample_df(filtered_bundle.test, sample_size=test_target, seed=args.seed)

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
        "source_filtered_sizes": source_filtered_sizes,
        "filtered_sizes": filtered_sizes,
        "train": dataset_basic_stats(train_sample),
        "valid": dataset_basic_stats(valid_sample),
        "test": dataset_basic_stats(test_sample),
        "target_total": args.target_size,
        "actual_total": len(train_sample) + len(valid_sample) + len(test_sample),
        "train_target": train_target,
        "valid_target": valid_target,
        "test_target": test_target,
        "top_k_labels": len(top_labels),
        "requested_top_k_labels": args.top_k_labels,
        "selected_top_k_labels": len(top_labels),
        "eligible_label_count": int(len(eligible_counts)),
        "min_label_support": args.min_label_support,
        "split_strategy": args.split_strategy,
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
    print(f"  fine labels  : {len(fine_labels)} (requested {args.top_k_labels})")
    print(f"  coarse labels: {len(coarse_labels)}")
    print(f"  split strategy: {args.split_strategy}")
    print(f"  min support   : {args.min_label_support}")
    print(f"  categories   : {', '.join(coarse_labels)}")


if __name__ == "__main__":
    main()
