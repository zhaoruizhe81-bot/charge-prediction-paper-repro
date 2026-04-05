"""数据读取、清洗和采样工具。"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import CHAPTER_RANGES, DEFAULT_RANDOM_SEED, PAPER_TOP_LEVEL_CATEGORIES, UNKNOWN_CATEGORY


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def anonymize_criminals(fact: str, criminals: list[str] | None) -> str:
    if not isinstance(fact, str):
        return ""
    text = fact
    if criminals:
        for name in criminals:
            if isinstance(name, str) and name:
                text = text.replace(name, "被告人")
    text = re.sub(r"(被告人\s*){2,}", "被告人", text)
    return text


def clean_fact_text(text: str) -> str:
    text = re.sub(r"[\r\n\t]+", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def extract_first_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        return int(match.group()) if match else None
    if isinstance(value, list):
        for item in value:
            int_value = extract_first_int(item)
            if int_value is not None:
                return int_value
    return None


def article_to_category(article_number: int | None) -> str:
    if article_number is None:
        return UNKNOWN_CATEGORY
    for start, end, category in CHAPTER_RANGES:
        if start <= article_number <= end:
            return category
    return UNKNOWN_CATEGORY


def normalize_accusation(accusation: str) -> str:
    if not isinstance(accusation, str):
        return ""
    return accusation.strip()


def parse_cail_split(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in read_jsonl(path):
        meta = record.get("meta", {})
        accusation_list = meta.get("accusation", []) or []
        accusation_list = [normalize_accusation(item) for item in accusation_list if normalize_accusation(item)]
        if not accusation_list:
            continue

        fact = anonymize_criminals(record.get("fact", ""), meta.get("criminals", []))
        fact = clean_fact_text(fact)
        if not fact:
            continue

        article_number = extract_first_int(meta.get("relevant_articles"))

        for accusation in accusation_list:
            rows.append(
                {
                    "fact": fact,
                    "accusation_list": accusation_list,
                    "article_number": article_number,
                    "fine_label": accusation,
                }
            )

    return pd.DataFrame(rows)


def parse_single_label_cail_split(
    path: str | Path,
    allowed_categories: list[str] | None = None,
) -> pd.DataFrame:
    allowed_set = set(allowed_categories or [])
    rows: list[dict[str, Any]] = []

    for record in read_jsonl(path):
        meta = record.get("meta", {})
        accusation_list = meta.get("accusation", []) or []
        accusation_list = [normalize_accusation(item) for item in accusation_list if normalize_accusation(item)]
        if len(accusation_list) != 1:
            continue

        fact = anonymize_criminals(record.get("fact", ""), meta.get("criminals", []))
        fact = clean_fact_text(fact)
        if not fact:
            continue

        article_number = extract_first_int(meta.get("relevant_articles"))
        coarse_label = article_to_category(article_number)
        if allowed_set and coarse_label not in allowed_set:
            continue

        accusation = accusation_list[0]
        rows.append(
            {
                "fact": fact,
                "accusation_list": accusation_list,
                "article_number": article_number,
                "fine_label": accusation,
                "coarse_label": coarse_label,
            }
        )

    return pd.DataFrame(rows)


def _compute_target_counts(label_counts: pd.Series, total_samples: int) -> dict[str, int]:
    total = int(label_counts.sum())
    if total_samples >= total:
        return {label: int(count) for label, count in label_counts.items()}

    ideal = (label_counts / total) * total_samples
    sampled = {label: int(value) for label, value in ideal.items()}

    if total_samples >= len(label_counts):
        for label, count in label_counts.items():
            if count > 0 and sampled[label] == 0:
                sampled[label] = 1

    allocated = sum(sampled.values())
    if allocated > total_samples:
        surplus = allocated - total_samples
        for label in sorted(sampled, key=sampled.get, reverse=True):
            if surplus == 0:
                break
            removable = min(surplus, max(0, sampled[label] - 1))
            sampled[label] -= removable
            surplus -= removable
    elif allocated < total_samples:
        shortage = total_samples - allocated
        fractional = sorted(
            ((ideal[label] - sampled[label], label) for label in sampled),
            reverse=True,
        )
        index = 0
        while shortage > 0 and fractional:
            _, label = fractional[index % len(fractional)]
            if sampled[label] < label_counts[label]:
                sampled[label] += 1
                shortage -= 1
            index += 1
            if index > total_samples * 5:
                break

    for label, count in sampled.items():
        if count > label_counts[label]:
            sampled[label] = int(label_counts[label])

    return sampled


def stratified_sample_df(
    dataframe: pd.DataFrame,
    sample_size: int,
    label_col: str = "fine_label",
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    if sample_size >= len(dataframe):
        return dataframe.copy().reset_index(drop=True)

    random.seed(seed)
    counts = dataframe[label_col].value_counts()
    target_counts = _compute_target_counts(counts, sample_size)

    sampled_parts: list[pd.DataFrame] = []
    for label, target in target_counts.items():
        group = dataframe[dataframe[label_col] == label]
        if target <= 0:
            continue
        if target >= len(group):
            sampled_parts.append(group)
        else:
            sampled_parts.append(group.sample(n=target, random_state=seed))

    sampled = pd.concat(sampled_parts, axis=0)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=seed)
    elif len(sampled) < sample_size:
        remaining = dataframe.drop(sampled.index)
        needed = sample_size - len(sampled)
        if needed > 0 and len(remaining) > 0:
            sampled = pd.concat(
                [sampled, remaining.sample(n=min(needed, len(remaining)), random_state=seed)],
                axis=0,
            )

    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled


def build_accusation_category_mapping(dataframe: pd.DataFrame) -> dict[str, str]:
    mapping_counter: dict[str, Counter[str]] = defaultdict(Counter)

    for _, row in dataframe.iterrows():
        accusation = row["fine_label"]
        category = article_to_category(row.get("article_number"))
        mapping_counter[accusation][category] += 1

    final_mapping: dict[str, str] = {}
    for accusation, category_counter in mapping_counter.items():
        final_mapping[accusation] = category_counter.most_common(1)[0][0]
    return final_mapping


def build_single_label_paper_dataset(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k_labels: int = 110,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], list[str]]:
    allowed_categories = set(PAPER_TOP_LEVEL_CATEGORIES)

    train_df = train_df[train_df["coarse_label"].isin(allowed_categories)].copy()
    valid_df = valid_df[valid_df["coarse_label"].isin(allowed_categories)].copy()
    test_df = test_df[test_df["coarse_label"].isin(allowed_categories)].copy()

    top_labels = train_df["fine_label"].value_counts().head(top_k_labels).index.tolist()
    top_label_set = set(top_labels)

    train_df = train_df[train_df["fine_label"].isin(top_label_set)].reset_index(drop=True)
    valid_df = valid_df[valid_df["fine_label"].isin(top_label_set)].reset_index(drop=True)
    test_df = test_df[test_df["fine_label"].isin(top_label_set)].reset_index(drop=True)

    accusation_to_category = (
        train_df[["fine_label", "coarse_label"]]
        .drop_duplicates()
        .set_index("fine_label")["coarse_label"]
        .to_dict()
    )
    return train_df, valid_df, test_df, accusation_to_category, top_labels


def select_top_fine_labels(
    dataframe: pd.DataFrame,
    *,
    top_k_labels: int = 110,
    min_label_support: int = 1,
) -> tuple[list[str], pd.Series]:
    if top_k_labels <= 0:
        raise ValueError("top_k_labels must be positive.")
    if min_label_support <= 0:
        raise ValueError("min_label_support must be positive.")

    counts = dataframe["fine_label"].value_counts()
    eligible_counts = counts[counts >= min_label_support]
    top_labels = eligible_counts.head(top_k_labels).index.tolist()
    return top_labels, eligible_counts


def build_accusation_to_category_from_df(dataframe: pd.DataFrame) -> dict[str, str]:
    if dataframe.empty:
        return {}
    return (
        dataframe[["fine_label", "coarse_label"]]
        .drop_duplicates()
        .set_index("fine_label")["coarse_label"]
        .to_dict()
    )


def _normalize_split_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> tuple[float, float, float]:
    total = float(train_ratio + valid_ratio + test_ratio)
    if total <= 0:
        raise ValueError("Split ratios must sum to a positive value.")
    return train_ratio / total, valid_ratio / total, test_ratio / total


def _allocate_group_split_counts(
    group_size: int,
    ratios: tuple[float, float, float],
    *,
    min_count_per_split: int = 1,
) -> tuple[int, int, int]:
    if group_size < min_count_per_split * 3:
        raise ValueError(
            f"Group size {group_size} is too small for 3-way split with min_count_per_split={min_count_per_split}."
        )

    train_ratio, valid_ratio, test_ratio = ratios
    counts = [min_count_per_split, min_count_per_split, min_count_per_split]
    remaining = group_size - sum(counts)
    if remaining <= 0:
        return counts[0], counts[1], counts[2]

    ideal = [
        train_ratio * remaining,
        valid_ratio * remaining,
        test_ratio * remaining,
    ]
    floors = [int(value) for value in ideal]
    counts = [base + floor for base, floor in zip(counts, floors)]

    leftover = remaining - sum(floors)
    order = sorted(
        range(3),
        key=lambda index: (ideal[index] - floors[index], ratios[index]),
        reverse=True,
    )
    for index in order[:leftover]:
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def rebuild_stratified_splits(
    dataframe: pd.DataFrame,
    *,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    label_col: str = "fine_label",
    seed: int = DEFAULT_RANDOM_SEED,
    min_count_per_split: int = 1,
) -> DatasetBundle:
    ratios = _normalize_split_ratios(train_ratio, valid_ratio, test_ratio)

    train_parts: list[pd.DataFrame] = []
    valid_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    label_counts = dataframe[label_col].value_counts()
    ordered_labels = label_counts.index.tolist()
    for index, label in enumerate(ordered_labels):
        group = dataframe[dataframe[label_col] == label].sample(frac=1.0, random_state=seed + index)
        train_count, valid_count, test_count = _allocate_group_split_counts(
            len(group),
            ratios,
            min_count_per_split=min_count_per_split,
        )

        train_parts.append(group.iloc[:train_count])
        valid_parts.append(group.iloc[train_count : train_count + valid_count])
        test_parts.append(group.iloc[train_count + valid_count : train_count + valid_count + test_count])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    valid_df = pd.concat(valid_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return DatasetBundle(train=train_df, valid=valid_df, test=test_df)


def apply_hierarchy_labels(
    dataframe: pd.DataFrame,
    accusation_to_category: dict[str, str],
) -> pd.DataFrame:
    result = dataframe.copy()
    result["coarse_label"] = result["fine_label"].map(accusation_to_category).fillna(UNKNOWN_CATEGORY)
    return result


def split_sample_sizes(
    train_size: int,
    valid_size: int,
    test_size: int,
    total_target: int,
) -> tuple[int, int, int]:
    total = train_size + valid_size + test_size
    train_target = round(total_target * train_size / total)
    valid_target = round(total_target * valid_size / total)
    test_target = total_target - train_target - valid_target
    return train_target, valid_target, test_target


def dataframe_to_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    records = dataframe.to_dict(orient="records")
    for item in records:
        if isinstance(item.get("accusation_list"), tuple):
            item["accusation_list"] = list(item["accusation_list"])
    return records


def dataset_basic_stats(dataframe: pd.DataFrame, label_col: str = "fine_label") -> dict[str, Any]:
    label_distribution = dataframe[label_col].value_counts().to_dict()
    text_length = dataframe["fact"].str.len()
    accusation_lengths = dataframe["accusation_list"].apply(lambda item: len(item) if isinstance(item, list) else 0)
    return {
        "num_samples": int(len(dataframe)),
        "num_labels": int(dataframe[label_col].nunique()),
        "avg_accusation_count": float(accusation_lengths.mean()) if len(accusation_lengths) else 0.0,
        "max_accusation_count": int(accusation_lengths.max()) if len(accusation_lengths) else 0,
        "avg_text_len": float(text_length.mean()),
        "median_text_len": float(text_length.median()),
        "max_text_len": int(text_length.max()),
        "min_text_len": int(text_length.min()),
        "top_20_labels": dict(list(label_distribution.items())[:20]),
    }


def export_processed_analysis_tables(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    """Export processed 50k dataset into CSV tables for visualization/statistics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    split_frames: list[pd.DataFrame] = []
    for split_name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        frame = split_df.copy()
        frame.insert(0, "split", split_name)
        frame["text_length"] = frame["fact"].astype(str).str.len()
        split_frames.append(frame)

    full_table = pd.concat(split_frames, axis=0, ignore_index=True)

    ordered_cols = [
        "split",
        "fact",
        "fine_label",
        "coarse_label",
        "article_number",
        "text_length",
        "accusation_list",
    ]
    available_cols = [col for col in ordered_cols if col in full_table.columns]
    full_table = full_table[available_cols]

    full_table_path = output_path / "processed_50k_table.csv"
    full_table.to_csv(full_table_path, index=False, encoding="utf-8-sig")

    fine_distribution = (
        full_table.groupby(["split", "fine_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    fine_distribution["ratio"] = (
        fine_distribution["count"]
        / fine_distribution.groupby("split")["count"].transform("sum").clip(lower=1)
    )
    fine_distribution["ratio"] = fine_distribution["ratio"].round(8)
    fine_distribution_path = output_path / "fine_label_distribution.csv"
    fine_distribution.to_csv(fine_distribution_path, index=False, encoding="utf-8-sig")

    coarse_distribution = (
        full_table.groupby(["split", "coarse_label"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    coarse_distribution["ratio"] = (
        coarse_distribution["count"]
        / coarse_distribution.groupby("split")["count"].transform("sum").clip(lower=1)
    )
    coarse_distribution["ratio"] = coarse_distribution["ratio"].round(8)
    coarse_distribution_path = output_path / "coarse_label_distribution.csv"
    coarse_distribution.to_csv(coarse_distribution_path, index=False, encoding="utf-8-sig")

    text_length_summary = (
        full_table.groupby("split")["text_length"]
        .agg(
            num_samples="size",
            avg_text_length="mean",
            median_text_length="median",
            p90_text_length=lambda series: series.quantile(0.9),
            min_text_length="min",
            max_text_length="max",
        )
        .reset_index()
    )
    text_length_summary["avg_text_length"] = text_length_summary["avg_text_length"].round(4)
    text_length_summary["median_text_length"] = text_length_summary["median_text_length"].round(4)
    text_length_summary["p90_text_length"] = text_length_summary["p90_text_length"].round(4)
    text_length_summary_path = output_path / "text_length_summary.csv"
    text_length_summary.to_csv(text_length_summary_path, index=False, encoding="utf-8-sig")

    return {
        "full_table": str(full_table_path),
        "fine_distribution": str(fine_distribution_path),
        "coarse_distribution": str(coarse_distribution_path),
        "text_length_summary": str(text_length_summary_path),
    }
