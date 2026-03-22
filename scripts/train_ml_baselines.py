#!/usr/bin/env python
"""训练 3 个传统机器学习基线模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.data_utils import read_jsonl
from charge_prediction.fusion import (
    hierarchical_constrained_decode,
    metric_delta,
    score_matrix_from_estimator,
    tune_hierarchical_fusion,
)
from charge_prediction.metrics import compute_classification_metrics
from charge_prediction.ml_models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML baselines for charge prediction")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ml_baselines"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svm", "sgd", "pa"],
        help="Model names: lr, svm, sgd, pa",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--max-valid-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--max-test-samples", type=int, default=0, help="0 means use all")
    parser.add_argument("--disable-hier-fusion", action="store_true", help="Disable hierarchical fusion")
    parser.add_argument(
        "--fusion-selection-metric",
        type=str,
        default="f1_macro",
        choices=["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
        help="Validation metric for selecting hierarchical fusion hyperparameters",
    )
    parser.add_argument(
        "--max-fusion-top-k-coarse",
        type=int,
        default=2,
        help="Use top-k coarse predictions when applying hierarchical constraints",
    )
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def limit_df(df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
    if max_samples > 0 and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df


def fit_label_encoder(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_series(series: pd.Series, label2id: dict[str, int]) -> np.ndarray:
    return series.map(label2id).fillna(-1).astype(int).to_numpy()


def build_coarse_to_fine_map(train_coarse: np.ndarray, train_fine: np.ndarray) -> dict[int, set[int]]:
    coarse_to_fine: dict[int, set[int]] = {}
    for coarse_id, fine_id in zip(train_coarse.tolist(), train_fine.tolist()):
        coarse_to_fine.setdefault(int(coarse_id), set()).add(int(fine_id))
    return coarse_to_fine


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.data_dir / "train_50k.jsonl")
    valid_df = load_split(args.data_dir / "valid_50k.jsonl")
    test_df = load_split(args.data_dir / "test_50k.jsonl")

    train_df = limit_df(train_df, args.max_train_samples)
    valid_df = limit_df(valid_df, args.max_valid_samples)
    test_df = limit_df(test_df, args.max_test_samples)

    fine_label2id, fine_id2label = fit_label_encoder(train_df["fine_label"].tolist())
    coarse_label2id, coarse_id2label = fit_label_encoder(train_df["coarse_label"].tolist())

    y_train_fine = encode_series(train_df["fine_label"], fine_label2id)
    y_valid_fine_all = encode_series(valid_df["fine_label"], fine_label2id)
    y_test_fine_all = encode_series(test_df["fine_label"], fine_label2id)

    valid_mask = y_valid_fine_all >= 0
    test_mask = y_test_fine_all >= 0

    y_valid_fine = y_valid_fine_all[valid_mask]
    y_test_fine = y_test_fine_all[test_mask]

    y_train_coarse = encode_series(train_df["coarse_label"], coarse_label2id)
    y_valid_coarse = encode_series(valid_df["coarse_label"], coarse_label2id)[valid_mask]
    y_test_coarse = encode_series(test_df["coarse_label"], coarse_label2id)[test_mask]

    if len(y_valid_fine) == 0 or len(y_test_fine) == 0:
        raise RuntimeError("No valid/test samples after label alignment.")

    coarse_to_fine = build_coarse_to_fine_map(y_train_coarse, y_train_fine)

    x_train = train_df["fact"].tolist()
    x_valid = valid_df.loc[valid_mask, "fact"].tolist()
    x_test = test_df.loc[test_mask, "fact"].tolist()

    metrics_summary: dict[str, dict[str, object]] = {}

    for model_name in args.models:
        print(f"\n[Training] {model_name}", flush=True)
        fine_pipeline = build_model(model_name, seed=args.seed)
        fine_pipeline.fit(x_train, y_train_fine)

        valid_pred = fine_pipeline.predict(x_valid)
        test_pred = fine_pipeline.predict(x_test)

        valid_metrics = compute_classification_metrics(y_valid_fine, valid_pred)
        test_metrics = compute_classification_metrics(y_test_fine, test_pred)

        model_path = args.output_dir / f"{model_name}.joblib"
        joblib.dump(
            {
                "pipeline": fine_pipeline,
                "label2id": fine_label2id,
                "id2label": fine_id2label,
            },
            model_path,
        )

        metrics_summary[model_name] = {
            "valid": valid_metrics,
            "test": test_metrics,
            "model_path": str(model_path),
            "num_fine_labels": len(fine_label2id),
        }

        print(f"[Valid] {valid_metrics}", flush=True)
        print(f"[Test ] {test_metrics}", flush=True)

        if args.disable_hier_fusion:
            continue

        coarse_pipeline = build_model(model_name, seed=args.seed)
        coarse_pipeline.fit(x_train, y_train_coarse)

        valid_fine_scores = score_matrix_from_estimator(fine_pipeline, x_valid, num_classes=len(fine_label2id))
        test_fine_scores = score_matrix_from_estimator(fine_pipeline, x_test, num_classes=len(fine_label2id))
        valid_coarse_scores = score_matrix_from_estimator(coarse_pipeline, x_valid, num_classes=len(coarse_label2id))
        test_coarse_scores = score_matrix_from_estimator(coarse_pipeline, x_test, num_classes=len(coarse_label2id))

        tuning = tune_hierarchical_fusion(
            y_true=y_valid_fine,
            fine_scores=valid_fine_scores,
            coarse_scores=valid_coarse_scores,
            coarse_to_fine=coarse_to_fine,
            metric_name=args.fusion_selection_metric,
            max_top_k_coarse=max(1, args.max_fusion_top_k_coarse),
        )
        fusion_config = tuning["best_config"]
        hier_valid_metrics = tuning["best_metrics"]

        if bool(fusion_config.get("use_hier_fusion", False)):
            hier_test_pred = hierarchical_constrained_decode(
                fine_scores=test_fine_scores,
                coarse_scores=test_coarse_scores,
                coarse_to_fine=coarse_to_fine,
                top_k_coarse=int(fusion_config.get("top_k_coarse", 1)),
                confidence_threshold=float(fusion_config.get("confidence_threshold", 0.0)),
                max_fine_margin=float(fusion_config.get("max_fine_margin", float("inf"))),
            )
        else:
            hier_test_pred = np.argmax(test_fine_scores, axis=1)

        hier_test_metrics = compute_classification_metrics(y_test_fine, hier_test_pred)

        coarse_valid_metrics: dict[str, float] = {}
        coarse_test_metrics: dict[str, float] = {}
        valid_coarse_mask = y_valid_coarse >= 0
        test_coarse_mask = y_test_coarse >= 0
        if np.any(valid_coarse_mask):
            valid_coarse_pred = np.argmax(valid_coarse_scores[valid_coarse_mask], axis=1)
            coarse_valid_metrics = compute_classification_metrics(y_valid_coarse[valid_coarse_mask], valid_coarse_pred)
        if np.any(test_coarse_mask):
            test_coarse_pred = np.argmax(test_coarse_scores[test_coarse_mask], axis=1)
            coarse_test_metrics = compute_classification_metrics(y_test_coarse[test_coarse_mask], test_coarse_pred)

        hier_model_path = args.output_dir / f"{model_name}_hierarchical.joblib"
        joblib.dump(
            {
                "fine_pipeline": fine_pipeline,
                "coarse_pipeline": coarse_pipeline,
                "fine_label2id": fine_label2id,
                "fine_id2label": fine_id2label,
                "coarse_label2id": coarse_label2id,
                "coarse_id2label": coarse_id2label,
                "coarse_to_fine": {k: sorted(v) for k, v in coarse_to_fine.items()},
                "fusion_config": fusion_config,
            },
            hier_model_path,
        )

        metrics_summary[model_name]["hierarchical_fusion"] = {
            "enabled": True,
            "selection_metric": tuning["metric_name"],
            "fusion_config": fusion_config,
            "valid": hier_valid_metrics,
            "test": hier_test_metrics,
            "delta_test_vs_base": metric_delta(hier_test_metrics, test_metrics),
            "coarse_valid": coarse_valid_metrics,
            "coarse_test": coarse_test_metrics,
            "model_path": str(hier_model_path),
        }

        print(f"[Hier Valid] {hier_valid_metrics}", flush=True)
        print(f"[Hier Test ] {hier_test_metrics}", flush=True)
        print(f"[Δ Test    ] {metric_delta(hier_test_metrics, test_metrics)}", flush=True)

    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[Done] ML baselines finished", flush=True)


if __name__ == "__main__":
    main()
