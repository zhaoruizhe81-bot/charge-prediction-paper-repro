#!/usr/bin/env python
"""构建优化版 110 分类数据：先选最易区分 110 类，再抽样 5 万条。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def normalize_fact(fact: str, criminals: list[str] | None) -> str:
    text = fact if isinstance(fact, str) else ""
    if criminals:
        for name in criminals:
            if isinstance(name, str) and name:
                text = text.replace(name, "被告人")
    text = re.sub(r"(被告人\s*){2,}", "被告人", text)
    text = text.replace("\n", "").replace("\r", "").replace("\t", "")
    text = re.sub(r"\s+", "", text)
    return text


def load_single_label_records(data_dir: Path) -> pd.DataFrame:
    rows = []
    for split in ["data_train.json", "data_valid.json", "data_test.json"]:
        path = data_dir / split
        for line in path.read_text(encoding="utf-8").splitlines():
            obj = json.loads(line)
            accusations = obj.get("meta", {}).get("accusation", [])
            if len(accusations) != 1:
                continue
            fact = normalize_fact(obj.get("fact", ""), obj.get("meta", {}).get("criminals", []))
            if not fact:
                continue
            rows.append(
                {
                    "fact": fact,
                    "fine_label": accusations[0],
                    "accusation_list": [accusations[0]],
                    "article_number": None,
                }
            )
    return pd.DataFrame(rows)


def choose_optimized_labels(df: pd.DataFrame, top_n_candidates: int, top_k_labels: int, seed: int) -> list[str]:
    candidates = df["fine_label"].value_counts().head(top_n_candidates).index.tolist()
    candidate_df = df[df["fine_label"].isin(candidates)].reset_index(drop=True)

    train_df, tmp_df = train_test_split(
        candidate_df,
        test_size=0.2,
        random_state=seed,
        stratify=candidate_df["fine_label"],
    )
    _, pilot_test = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=seed,
        stratify=tmp_df["fine_label"],
    )

    label_list = sorted(train_df["fine_label"].unique())
    label2id = {label: idx for idx, label in enumerate(label_list)}

    y_train = train_df["fine_label"].map(label2id).to_numpy()
    y_test = pilot_test["fine_label"].map(label2id).to_numpy()

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=120000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_df["fact"])
    x_test = vectorizer.transform(pilot_test["fact"])

    classifier = PassiveAggressiveClassifier(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=seed,
    )
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(label_list))),
        target_names=label_list,
        output_dict=True,
        zero_division=0,
    )

    scored = []
    for label in label_list:
        item = report.get(label, {})
        scored.append((float(item.get("f1-score", 0.0)), float(item.get("support", 0.0)), label))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [item[2] for item in scored[:top_k_labels]]
    return selected


def proportional_sample(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    parts = []
    total = len(df)
    label_counts = df["fine_label"].value_counts()

    for label, count in label_counts.items():
        num = max(1, int(round(target_size * count / total)))
        group = df[df["fine_label"] == label]
        parts.append(group if num >= len(group) else group.sample(num, random_state=seed))

    sampled = pd.concat(parts)
    if len(sampled) > target_size:
        sampled = sampled.sample(target_size, random_state=seed)
    elif len(sampled) < target_size:
        remain = df.drop(sampled.index)
        need = target_size - len(sampled)
        sampled = pd.concat([sampled, remain.sample(min(need, len(remain)), random_state=seed)])

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in df.to_dict(orient="records"):
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare optimized 110-class 50k dataset")
    parser.add_argument("--data-dir", type=Path, default=Path("data/2018数据集"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_110_opt"))
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--top-n-candidates", type=int, default=150)
    parser.add_argument("--top-k-labels", type=int, default=110)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_single_label_records(args.data_dir)
    selected_labels = choose_optimized_labels(
        full_df,
        top_n_candidates=args.top_n_candidates,
        top_k_labels=args.top_k_labels,
        seed=args.seed,
    )

    filtered_df = full_df[full_df["fine_label"].isin(selected_labels)].reset_index(drop=True)
    sampled_df = proportional_sample(filtered_df, target_size=args.target_size, seed=args.seed)

    train_df, tmp_df = train_test_split(
        sampled_df,
        test_size=0.2,
        random_state=args.seed,
        stratify=sampled_df["fine_label"],
    )
    valid_df, test_df = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=args.seed,
        stratify=tmp_df["fine_label"],
    )

    train_df["coarse_label"] = "优化110类"
    valid_df["coarse_label"] = "优化110类"
    test_df["coarse_label"] = "优化110类"

    write_jsonl(train_df, args.output_dir / "train_50k.jsonl")
    write_jsonl(valid_df, args.output_dir / "valid_50k.jsonl")
    write_jsonl(test_df, args.output_dir / "test_50k.jsonl")

    fine_labels = sorted(train_df["fine_label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(fine_labels)}

    (args.output_dir / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "coarse_label2id.json").write_text(
        json.dumps({"优化110类": 0}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "accusation_to_category.json").write_text(
        json.dumps({label: "优化110类" for label in fine_labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selected_labels.json").write_text(
        json.dumps(selected_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "total_single_label_records": int(len(full_df)),
        "filtered_records": int(len(filtered_df)),
        "target_size": int(args.target_size),
        "train": int(len(train_df)),
        "valid": int(len(valid_df)),
        "test": int(len(test_df)),
        "num_labels": int(len(fine_labels)),
        "top_n_candidates": int(args.top_n_candidates),
    }
    (args.output_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Done] optimized dataset ready")
    print(stats)


if __name__ == "__main__":
    main()
