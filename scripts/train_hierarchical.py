#!/usr/bin/env python
"""训练层次分类模型（coarse -> fine）。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from charge_prediction.data_utils import read_jsonl
from charge_prediction.hierarchical import HierarchicalChargeClassifier
from charge_prediction.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hierarchical charge classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hierarchical"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def fit_label_encoder(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def transform_labels(series: pd.Series, label2id: dict[str, int]) -> np.ndarray:
    return series.map(label2id).fillna(-1).astype(int).to_numpy()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.data_dir / "train_50k.jsonl")
    valid_df = load_split(args.data_dir / "valid_50k.jsonl")
    test_df = load_split(args.data_dir / "test_50k.jsonl")

    fine_label2id, fine_id2label = fit_label_encoder(train_df["fine_label"].tolist())
    coarse_label2id, coarse_id2label = fit_label_encoder(train_df["coarse_label"].tolist())

    train_fine = transform_labels(train_df["fine_label"], fine_label2id)
    train_coarse = transform_labels(train_df["coarse_label"], coarse_label2id)

    valid_fine = transform_labels(valid_df["fine_label"], fine_label2id)
    valid_coarse = transform_labels(valid_df["coarse_label"], coarse_label2id)
    test_fine = transform_labels(test_df["fine_label"], fine_label2id)
    test_coarse = transform_labels(test_df["coarse_label"], coarse_label2id)

    valid_mask = (valid_fine >= 0) & (valid_coarse >= 0)
    test_mask = (test_fine >= 0) & (test_coarse >= 0)

    classifier = HierarchicalChargeClassifier(seed=args.seed)
    classifier.fit(
        texts=train_df["fact"].tolist(),
        coarse_labels=train_coarse,
        fine_labels=train_fine,
    )

    valid_coarse_pred, valid_fine_pred = classifier.predict(valid_df.loc[valid_mask, "fact"].tolist())
    test_coarse_pred, test_fine_pred = classifier.predict(test_df.loc[test_mask, "fact"].tolist())

    valid_coarse_metrics = compute_classification_metrics(valid_coarse[valid_mask], valid_coarse_pred)
    valid_fine_metrics = compute_classification_metrics(valid_fine[valid_mask], valid_fine_pred)

    test_coarse_metrics = compute_classification_metrics(test_coarse[test_mask], test_coarse_pred)
    test_fine_metrics = compute_classification_metrics(test_fine[test_mask], test_fine_pred)

    metrics = {
        "valid": {
            "coarse": valid_coarse_metrics,
            "fine": valid_fine_metrics,
        },
        "test": {
            "coarse": test_coarse_metrics,
            "fine": test_fine_metrics,
        },
        "coarse_num_labels": len(coarse_label2id),
        "fine_num_labels": len(fine_label2id),
    }

    joblib.dump(
        {
            "classifier": classifier,
            "fine_label2id": fine_label2id,
            "fine_id2label": fine_id2label,
            "coarse_label2id": coarse_label2id,
            "coarse_id2label": coarse_id2label,
        },
        args.output_dir / "hierarchical.joblib",
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Done] Hierarchical model trained")
    print("[Valid coarse]", valid_coarse_metrics)
    print("[Valid fine ]", valid_fine_metrics)
    print("[Test coarse ]", test_coarse_metrics)
    print("[Test fine  ]", test_fine_metrics)


if __name__ == "__main__":
    main()
