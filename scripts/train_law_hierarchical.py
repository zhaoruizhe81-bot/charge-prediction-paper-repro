#!/usr/bin/env python
"""Build hierarchical-fusion law recommendation results from flat model outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.metrics import (
    compute_classification_metrics,
    compute_multilabel_metrics,
    compute_multilabel_per_label_metrics,
    multilabel_predictions_from_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hierarchical fusion for law recommendation")
    parser.add_argument("--law-deep-dir", type=Path, default=Path("outputs_paper/law_deep"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper/law_hierarchical"))
    parser.add_argument("--models", nargs="+", default=["fc", "rcnn"])
    parser.add_argument("--base-model", type=str, default="auto")
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--weights", nargs="*", default=[])
    parser.add_argument(
        "--confusing-rule",
        type=str,
        default="below_mean_f1_or_recall",
        choices=["below_mean_f1_or_recall", "below_mean_f1", "all"],
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="f1_score",
        choices=["accuracy", "recall_macro", "recall_micro", "f1_macro", "f1_micro", "f1_score"],
    )
    parser.add_argument(
        "--selection-objective",
        type=str,
        default="accuracy_f1_recall",
        choices=["metric", "accuracy_f1_recall"],
    )
    return parser.parse_args()


def load_model_output(model_dir: Path) -> dict[str, np.ndarray]:
    path = model_dir / "eval_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing model eval output: {path}")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def parse_weight_candidates(raw_weights: list[str], num_models: int) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    if not raw_weights and num_models == 1:
        raw_weights = ["1.0"]
    elif not raw_weights and num_models == 2:
        raw_weights = [f"{w:.1f},{1.0 - w:.1f}" for w in np.linspace(0.0, 1.0, 11)]
    for item in raw_weights:
        values = [float(value.strip()) for value in item.split(",") if value.strip()]
        if len(values) != num_models:
            continue
        arr = np.asarray(values, dtype=np.float64)
        if arr.sum() <= 0:
            continue
        candidates.append(arr / arr.sum())
    if not candidates:
        candidates.append(np.ones(num_models, dtype=np.float64) / max(num_models, 1))
    return candidates


def objective_score(metrics: dict[str, float], base_metrics: dict[str, float], args: argparse.Namespace) -> tuple[int, float, float, float]:
    if args.selection_objective == "metric":
        return (
            1,
            float(metrics.get(args.selection_metric, 0.0)),
            float(metrics.get("accuracy", 0.0)),
            float(metrics.get("recall_macro", 0.0)),
        )
    keeps_accuracy = int(float(metrics.get("accuracy", 0.0)) >= float(base_metrics.get("accuracy", 0.0)) - 1e-12)
    return (
        keeps_accuracy,
        float(metrics.get("f1_score", 0.0)),
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("recall_macro", 0.0)),
    )


def choose_base_model(law_metrics: dict[str, object], requested: str) -> str:
    if requested != "auto":
        return requested
    best_model = ""
    best_score = (-1.0, -1.0)
    for model_name, payload in law_metrics.items():
        metrics = payload.get("tuned_valid") or payload.get("best_valid") or payload.get("test", {})
        score = (float(metrics.get("f1_score", 0.0)), float(metrics.get("accuracy", 0.0)))
        if score > best_score:
            best_score = score
            best_model = str(model_name)
    if not best_model:
        raise ValueError("Cannot infer --base-model auto because law_deep metrics are empty.")
    return best_model


def find_confusing_label_ids(y_valid: np.ndarray, valid_pred: np.ndarray, rule: str) -> list[int]:
    if rule == "all":
        return list(range(y_valid.shape[1]))
    rows = compute_multilabel_per_label_metrics(y_valid, valid_pred)
    mean_f1 = float(np.mean([float(row["f1"]) for row in rows])) if rows else 0.0
    mean_recall = float(np.mean([float(row["recall"]) for row in rows])) if rows else 0.0
    confusing: list[int] = []
    for row in rows:
        is_confusing = float(row["f1"]) < mean_f1
        if rule == "below_mean_f1_or_recall":
            is_confusing = is_confusing or float(row["recall"]) < mean_recall
        if is_confusing:
            confusing.append(int(row["label_id"]))
    return confusing


def fuse_scores(score_list: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    stacked = np.stack(score_list, axis=0)
    return np.tensordot(weights, stacked, axes=(0, 0))


def apply_hierarchical_prediction(
    base_scores: np.ndarray,
    fused_scores: np.ndarray,
    confusing_ids: list[int],
    *,
    base_threshold: float,
    confusing_threshold: float,
) -> np.ndarray:
    pred = multilabel_predictions_from_scores(base_scores, base_threshold)
    if confusing_ids:
        confusing_pred = (fused_scores[:, confusing_ids] >= confusing_threshold).astype(int)
        pred[:, confusing_ids] = confusing_pred
        empty_rows = np.where(pred.sum(axis=1) == 0)[0]
        if empty_rows.size:
            pred[empty_rows, np.argmax(fused_scores[empty_rows], axis=1)] = 1
    return pred


def binary_confusing_metrics(y_true: np.ndarray, y_pred: np.ndarray, confusing_ids: list[int]) -> dict[str, float]:
    if not confusing_ids:
        zeros = np.zeros(y_true.shape[0], dtype=int)
        return compute_classification_metrics(zeros, zeros)
    true_binary = (y_true[:, confusing_ids].sum(axis=1) > 0).astype(int)
    pred_binary = (y_pred[:, confusing_ids].sum(axis=1) > 0).astype(int)
    return compute_classification_metrics(true_binary, pred_binary)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    law_metrics = json.loads((args.law_deep_dir / "metrics.json").read_text(encoding="utf-8"))
    base_model = choose_base_model(law_metrics, args.base_model)
    outputs = {model: load_model_output(args.law_deep_dir / model) for model in args.models}
    if base_model not in outputs:
        raise ValueError(f"base model {base_model!r} not found in --models")
    base = outputs[base_model]
    y_valid = base["y_valid"]
    y_test = base["y_test"]
    base_valid_scores = base["valid_scores"]
    base_test_scores = base["test_scores"]

    base_metrics_path = args.law_deep_dir / base_model / "model_bundle.json"
    id2label: dict[int, str] = {}
    if base_metrics_path.exists():
        bundle = json.loads(base_metrics_path.read_text(encoding="utf-8"))
        id2label = {int(k): str(v) for k, v in bundle.get("id2label", {}).items()}

    base_threshold = float(law_metrics[base_model]["threshold"])
    base_valid_pred = multilabel_predictions_from_scores(base_valid_scores, base_threshold)
    base_test_pred = multilabel_predictions_from_scores(base_test_scores, base_threshold)
    base_valid_metrics = compute_multilabel_metrics(y_valid, base_valid_pred)
    base_test_metrics = compute_multilabel_metrics(y_test, base_test_pred)
    confusing_ids = find_confusing_label_ids(y_valid, base_valid_pred, args.confusing_rule)

    valid_scores_list = [outputs[model]["valid_scores"] for model in args.models]
    test_scores_list = [outputs[model]["test_scores"] for model in args.models]
    weight_candidates = parse_weight_candidates(args.weights, len(args.models))

    best: dict[str, object] | None = None
    best_accuracy: dict[str, object] | None = None
    best_f1: dict[str, object] | None = None
    for weights in weight_candidates:
        fused_valid_scores = fuse_scores(valid_scores_list, weights)
        for threshold in args.thresholds:
            valid_pred = apply_hierarchical_prediction(
                base_valid_scores,
                fused_valid_scores,
                confusing_ids,
                base_threshold=base_threshold,
                confusing_threshold=float(threshold),
            )
            metrics = compute_multilabel_metrics(y_valid, valid_pred)
            score = objective_score(metrics, base_valid_metrics, args)
            candidate = {
                "weights": weights,
                "confusing_threshold": float(threshold),
                "valid": metrics,
                "valid_pred": valid_pred,
                "fused_valid_scores": fused_valid_scores,
                "score": score,
            }
            best_score = best.get("score") if best is not None else None
            if best is None or score > best_score:
                best = candidate
            if best_accuracy is None or (
                float(metrics["accuracy"]),
                float(metrics["f1_score"]),
                float(metrics["recall_macro"]),
            ) > (
                float(best_accuracy["valid"]["accuracy"]),
                float(best_accuracy["valid"]["f1_score"]),
                float(best_accuracy["valid"]["recall_macro"]),
            ):
                best_accuracy = candidate
            if best_f1 is None or (
                float(metrics["f1_score"]),
                float(metrics["accuracy"]),
                float(metrics["recall_macro"]),
            ) > (
                float(best_f1["valid"]["f1_score"]),
                float(best_f1["valid"]["accuracy"]),
                float(best_f1["valid"]["recall_macro"]),
            ):
                best_f1 = candidate

    if best is None:
        raise RuntimeError("No hierarchical fusion candidate was evaluated.")

    def evaluate_candidate(candidate: dict[str, object]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        fused_scores = fuse_scores(test_scores_list, np.asarray(candidate["weights"], dtype=np.float64))
        pred = apply_hierarchical_prediction(
            base_test_scores,
            fused_scores,
            confusing_ids,
            base_threshold=base_threshold,
            confusing_threshold=float(candidate["confusing_threshold"]),
        )
        return fused_scores, pred, compute_multilabel_metrics(y_test, pred)

    fused_test_scores, test_pred, test_metrics = evaluate_candidate(best)
    accuracy_test_scores, accuracy_test_pred, accuracy_test_metrics = evaluate_candidate(best_accuracy or best)
    f1_test_scores, f1_test_pred, f1_test_metrics = evaluate_candidate(best_f1 or best)

    intermediate_rows = [
        {"split": "valid", "stage": "flat_law", **base_valid_metrics},
        {"split": "valid", "stage": "first_layer_confusing", **binary_confusing_metrics(y_valid, base_valid_pred, confusing_ids)},
        {"split": "valid", "stage": "final_hier_fusion", **best["valid"]},
        {"split": "valid", "stage": "final_hier_accuracy", **(best_accuracy or best)["valid"]},
        {"split": "valid", "stage": "final_hier_f1", **(best_f1 or best)["valid"]},
        {"split": "test", "stage": "flat_law", **base_test_metrics},
        {"split": "test", "stage": "first_layer_confusing", **binary_confusing_metrics(y_test, base_test_pred, confusing_ids)},
        {"split": "test", "stage": "final_hier_fusion", **test_metrics},
        {"split": "test", "stage": "final_hier_accuracy", **accuracy_test_metrics},
        {"split": "test", "stage": "final_hier_f1", **f1_test_metrics},
    ]
    pd.DataFrame(intermediate_rows).to_csv(args.output_dir / "intermediate_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(compute_multilabel_per_label_metrics(y_test, test_pred, id2label=id2label)).to_csv(
        args.output_dir / "per_label_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        args.output_dir / "eval_outputs.npz",
        y_valid=y_valid.astype(np.int8),
        valid_scores=np.asarray(best["fused_valid_scores"], dtype=np.float32),
        valid_pred=np.asarray(best["valid_pred"], dtype=np.int8),
        y_test=y_test.astype(np.int8),
        test_scores=fused_test_scores.astype(np.float32),
        test_pred=test_pred.astype(np.int8),
    )

    metrics = {
        "artifact_type": "law_hierarchical_fusion",
        "task_mode": "law_article_multilabel",
        "config": {
            "models": args.models,
            "base_model": base_model,
            "base_threshold": base_threshold,
            "confusing_rule": args.confusing_rule,
            "confusing_label_count": len(confusing_ids),
            "confusing_articles": [id2label.get(label_id, str(label_id)) for label_id in confusing_ids],
            "weights": [float(item) for item in np.asarray(best["weights"], dtype=float).tolist()],
            "confusing_threshold": float(best["confusing_threshold"]),
            "selection_metric": args.selection_metric,
            "selection_objective": args.selection_objective,
        },
        "valid": {
            "flat_law": base_valid_metrics,
            "fine_hier": best["valid"],
        },
        "test": {
            "flat_law": base_test_metrics,
            "fine_hier": test_metrics,
            "fine_hier_accuracy": accuracy_test_metrics,
            "fine_hier_f1": f1_test_metrics,
        },
        "intermediate_rows": intermediate_rows,
        "variant_configs": {
            "fine_hier": {
                "weights": [float(item) for item in np.asarray(best["weights"], dtype=float).tolist()],
                "confusing_threshold": float(best["confusing_threshold"]),
                "valid": best["valid"],
            },
            "fine_hier_accuracy": {
                "weights": [float(item) for item in np.asarray((best_accuracy or best)["weights"], dtype=float).tolist()],
                "confusing_threshold": float((best_accuracy or best)["confusing_threshold"]),
                "valid": (best_accuracy or best)["valid"],
            },
            "fine_hier_f1": {
                "weights": [float(item) for item in np.asarray((best_f1 or best)["weights"], dtype=float).tolist()],
                "confusing_threshold": float((best_f1 or best)["confusing_threshold"]),
                "valid": (best_f1 or best)["valid"],
            },
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[Done] law hierarchical fusion finished")
    print("[Test flat]", base_test_metrics)
    print("[Test hier]", test_metrics)


if __name__ == "__main__":
    main()
