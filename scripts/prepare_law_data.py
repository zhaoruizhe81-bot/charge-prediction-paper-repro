#!/usr/bin/env python
"""Prepare multi-label criminal-law article recommendation data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.constants import DEFAULT_RANDOM_SEED
from charge_prediction.data_utils import (
    DatasetBundle,
    dataframe_to_records,
    dataset_basic_stats,
    filter_rows_by_labels,
    parse_law_article_split,
    rebuild_stratified_splits,
    split_sample_sizes,
    stratified_sample_df,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CAIL law-article recommendation data")
    parser.add_argument("--data-dir", type=Path, default=Path("../2018数据集/2018数据集"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_law"))
    parser.add_argument("--target-size", type=int, default=0, help="0 means keep all filtered rows")
    parser.add_argument("--top-k-articles", type=int, default=183)
    parser.add_argument("--min-label-support", type=int, default=3)
    parser.add_argument(
        "--split-strategy",
        type=str,
        default="rebuild_stable",
        choices=["rebuild_stable", "official_filtered"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def save_split(dataframe: pd.DataFrame, path: Path) -> None:
    write_jsonl(dataframe_to_records(dataframe), path)


def multilabel_stats(dataframe: pd.DataFrame, label_col: str = "article_numbers") -> dict[str, object]:
    if dataframe.empty:
        return {"num_samples": 0, "num_labels": 0, "avg_label_count": 0.0, "max_label_count": 0}
    exploded = dataframe[label_col].explode()
    label_distribution = exploded.value_counts().head(20).to_dict()
    label_counts = dataframe[label_col].apply(len)
    text_length = dataframe["fact"].astype(str).str.len()
    return {
        "num_samples": int(len(dataframe)),
        "num_labels": int(exploded.nunique()),
        "avg_label_count": float(label_counts.mean()),
        "max_label_count": int(label_counts.max()),
        "avg_text_len": float(text_length.mean()),
        "median_text_len": float(text_length.median()),
        "max_text_len": int(text_length.max()),
        "min_text_len": int(text_length.min()),
        "top_20_labels": {str(int(k)): int(v) for k, v in label_distribution.items()},
    }


def export_analysis_tables(bundle: DatasetBundle, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for split, df in [("train", bundle.train), ("valid", bundle.valid), ("test", bundle.test)]:
        frame = df.copy()
        frame.insert(0, "split", split)
        frame["text_length"] = frame["fact"].astype(str).str.len()
        frame["article_numbers_text"] = frame["article_numbers"].apply(lambda labels: ",".join(str(item) for item in labels))
        frames.append(frame)
    full = pd.concat(frames, axis=0, ignore_index=True)
    full_path = output_dir / "law_processed_table.csv"
    full[["split", "fact", "article_numbers_text", "primary_article", "article_count", "text_length"]].to_csv(
        full_path,
        index=False,
        encoding="utf-8-sig",
    )

    dist = (
        full[["split", "article_numbers"]]
        .explode("article_numbers")
        .groupby(["split", "article_numbers"])
        .size()
        .reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    dist_path = output_dir / "law_article_distribution.csv"
    dist.to_csv(dist_path, index=False, encoding="utf-8-sig")
    return {"full_table": str(full_path), "article_distribution": str(dist_path)}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = parse_law_article_split(args.data_dir / "data_train.json")
    valid_df = parse_law_article_split(args.data_dir / "data_valid.json")
    test_df = parse_law_article_split(args.data_dir / "data_test.json")
    pooled = pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)

    article_counts = pooled["article_numbers"].explode().value_counts()
    eligible = article_counts[article_counts >= args.min_label_support]
    selected_articles = [int(item) for item in eligible.head(args.top_k_articles).index.tolist()]
    selected_set = set(selected_articles)
    if not selected_articles:
        raise ValueError("No law articles matched the requested support threshold.")

    official_bundle = DatasetBundle(
        train=filter_rows_by_labels(train_df, selected_set),
        valid=filter_rows_by_labels(valid_df, selected_set),
        test=filter_rows_by_labels(test_df, selected_set),
    )
    if args.split_strategy == "official_filtered":
        bundle = official_bundle
    else:
        pooled_selected = filter_rows_by_labels(pooled, selected_set)
        primary_counts = pooled_selected["primary_article"].value_counts()
        too_small = primary_counts[primary_counts < 3]
        if not too_small.empty:
            pooled_selected = pooled_selected[~pooled_selected["primary_article"].isin(too_small.index)].reset_index(drop=True)
            selected_set = set(int(item) for item in pooled_selected["article_numbers"].explode().dropna().unique().tolist())
            selected_articles = sorted(selected_set)
        bundle = rebuild_stratified_splits(
            pooled_selected,
            train_ratio=len(official_bundle.train),
            valid_ratio=len(official_bundle.valid),
            test_ratio=len(official_bundle.test),
            label_col="primary_article",
            seed=args.seed,
            min_count_per_split=1,
        )

    if args.target_size > 0:
        train_target, valid_target, test_target = split_sample_sizes(
            len(bundle.train),
            len(bundle.valid),
            len(bundle.test),
            args.target_size,
        )
        bundle = DatasetBundle(
            train=stratified_sample_df(bundle.train, train_target, label_col="primary_article", seed=args.seed),
            valid=stratified_sample_df(bundle.valid, valid_target, label_col="primary_article", seed=args.seed),
            test=stratified_sample_df(bundle.test, test_target, label_col="primary_article", seed=args.seed),
        )

    article_labels = sorted(int(item) for item in selected_set)
    label2id = {str(label): index for index, label in enumerate(article_labels)}
    id2label = {str(index): str(label) for label, index in label2id.items()}

    save_split(bundle.train, args.output_dir / "train.jsonl")
    save_split(bundle.valid, args.output_dir / "valid.jsonl")
    save_split(bundle.test, args.output_dir / "test.jsonl")
    (args.output_dir / "label2id.json").write_text(json.dumps(label2id, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "id2label.json").write_text(json.dumps(id2label, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "top_articles.json").write_text(
        json.dumps(article_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "task_mode": "law_article_multilabel",
        "split_strategy": args.split_strategy,
        "target_size": args.target_size,
        "top_k_articles": len(article_labels),
        "requested_top_k_articles": args.top_k_articles,
        "min_label_support": args.min_label_support,
        "raw_sizes": {"train": len(train_df), "valid": len(valid_df), "test": len(test_df)},
        "train": multilabel_stats(bundle.train),
        "valid": multilabel_stats(bundle.valid),
        "test": multilabel_stats(bundle.test),
        "analysis_tables": export_analysis_tables(bundle, args.output_dir / "analysis"),
    }
    (args.output_dir / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] law-article recommendation data prepared")
    print(f"  train: {len(bundle.train)}")
    print(f"  valid: {len(bundle.valid)}")
    print(f"  test : {len(bundle.test)}")
    print(f"  law labels: {len(article_labels)}")
    print(f"  split strategy: {args.split_strategy}")


if __name__ == "__main__":
    main()
