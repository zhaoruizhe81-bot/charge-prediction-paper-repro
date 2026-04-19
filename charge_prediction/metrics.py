"""评估指标。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def _per_class_stats(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tp = np.zeros(len(labels), dtype=np.float64)
    fp = np.zeros(len(labels), dtype=np.float64)
    fn = np.zeros(len(labels), dtype=np.float64)

    for idx, label in enumerate(labels.tolist()):
        true_mask = y_true == label
        pred_mask = y_pred == label
        tp[idx] = float(np.sum(true_mask & pred_mask))
        fp[idx] = float(np.sum(~true_mask & pred_mask))
        fn[idx] = float(np.sum(true_mask & ~pred_mask))

    return tp, fp, fn


def _f1_from_stats(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    return np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape[0] == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "precision_micro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_micro": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
            "f1_score": 0.0,
        }

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    accuracy = float(np.mean(y_true == y_pred))
    labels = np.unique(np.concatenate([y_true, y_pred]))
    tp, fp, fn = _per_class_stats(y_true, y_pred, labels)
    precision_per_class = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall_per_class = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1_per_class = _f1_from_stats(tp, fp, fn)

    support = np.array([(y_true == label).sum() for label in labels], dtype=np.float64)
    total_support = float(np.sum(support))

    macro_precision = float(np.mean(precision_per_class)) if precision_per_class.size else 0.0
    macro_recall = float(np.mean(recall_per_class)) if recall_per_class.size else 0.0
    macro_f1 = float(np.mean(f1_per_class)) if f1_per_class.size else 0.0
    weighted_precision = float(np.sum(precision_per_class * support) / total_support) if total_support > 0 else 0.0
    weighted_recall = float(np.sum(recall_per_class * support) / total_support) if total_support > 0 else 0.0
    weighted_f1 = float(np.sum(f1_per_class * support) / total_support) if total_support > 0 else 0.0

    tp_total = float(np.sum(tp))
    fp_total = float(np.sum(fp))
    fn_total = float(np.sum(fn))
    precision_micro = _safe_div(tp_total, tp_total + fp_total)
    recall_micro = _safe_div(tp_total, tp_total + fn_total)
    micro_f1 = _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)

    return {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "precision_micro": precision_micro,
        "precision_weighted": weighted_precision,
        "recall_macro": macro_recall,
        "recall_micro": recall_micro,
        "recall_weighted": weighted_recall,
        "f1_macro": macro_f1,
        "f1_micro": micro_f1,
        "f1_weighted": weighted_f1,
        "f1_score": float((macro_f1 + micro_f1) / 2.0),
    }


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute paper-style metrics for binary multi-label targets."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return {
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "precision_micro": 0.0,
            "precision_weighted": 0.0,
            "recall_macro": 0.0,
            "recall_micro": 0.0,
            "recall_weighted": 0.0,
            "f1_macro": 0.0,
            "f1_micro": 0.0,
            "f1_weighted": 0.0,
            "f1_score": 0.0,
        }
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if y_true.ndim != 2:
        raise ValueError("multi-label metrics expect a 2D indicator matrix.")

    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))
    tp = np.sum((y_true == 1) & (y_pred == 1), axis=0).astype(np.float64)
    fp = np.sum((y_true == 0) & (y_pred == 1), axis=0).astype(np.float64)
    fn = np.sum((y_true == 1) & (y_pred == 0), axis=0).astype(np.float64)
    support = np.sum(y_true == 1, axis=0).astype(np.float64)
    total_support = float(np.sum(support))

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    tp_total = float(np.sum(tp))
    fp_total = float(np.sum(fp))
    fn_total = float(np.sum(fn))
    precision_micro = _safe_div(tp_total, tp_total + fp_total)
    recall_micro = _safe_div(tp_total, tp_total + fn_total)
    f1_micro = _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro)

    macro_precision = float(np.mean(precision)) if precision.size else 0.0
    macro_recall = float(np.mean(recall)) if recall.size else 0.0
    macro_f1 = float(np.mean(f1)) if f1.size else 0.0
    weighted_precision = float(np.sum(precision * support) / total_support) if total_support > 0 else 0.0
    weighted_recall = float(np.sum(recall * support) / total_support) if total_support > 0 else 0.0
    weighted_f1 = float(np.sum(f1 * support) / total_support) if total_support > 0 else 0.0

    return {
        "accuracy": exact_match,
        "precision_macro": macro_precision,
        "precision_micro": precision_micro,
        "precision_weighted": weighted_precision,
        "recall_macro": macro_recall,
        "recall_micro": recall_micro,
        "recall_weighted": weighted_recall,
        "f1_macro": macro_f1,
        "f1_micro": f1_micro,
        "f1_weighted": weighted_f1,
        "f1_score": float((macro_f1 + f1_micro) / 2.0),
    }


def compute_multilabel_per_label_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    id2label: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    rows: list[dict[str, Any]] = []
    for label_id in range(y_true.shape[1]):
        true_col = y_true[:, label_id]
        pred_col = y_pred[:, label_id]
        tp = float(np.sum((true_col == 1) & (pred_col == 1)))
        fp = float(np.sum((true_col == 0) & (pred_col == 1)))
        fn = float(np.sum((true_col == 1) & (pred_col == 0)))
        support = int(np.sum(true_col == 1))
        pred_support = int(np.sum(pred_col == 1))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        rows.append(
            {
                "label_id": int(label_id),
                "label": (id2label or {}).get(int(label_id), str(label_id)),
                "support": support,
                "pred_support": pred_support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def multilabel_predictions_from_scores(scores: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    if pred.ndim == 2 and pred.shape[0] > 0:
        empty_rows = np.where(pred.sum(axis=1) == 0)[0]
        if empty_rows.size:
            pred[empty_rows, np.argmax(scores[empty_rows], axis=1)] = 1
    return pred


def tune_multilabel_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric_name: str = "f1_score",
    thresholds: list[float] | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = thresholds or [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    best_threshold = float(candidates[0])
    best_metrics = compute_multilabel_metrics(y_true, multilabel_predictions_from_scores(scores, best_threshold))
    for threshold in candidates[1:]:
        metrics = compute_multilabel_metrics(y_true, multilabel_predictions_from_scores(scores, float(threshold)))
        if float(metrics.get(metric_name, 0.0)) > float(best_metrics.get(metric_name, 0.0)):
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    label_ids: list[int] | np.ndarray | None = None,
    id2label: dict[int, str] | None = None,
    train_support: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if label_ids is None:
        labels = np.unique(np.concatenate([y_true, y_pred])) if y_true.size or y_pred.size else np.asarray([], dtype=int)
    else:
        labels = np.asarray(label_ids, dtype=int)

    rows: list[dict[str, Any]] = []
    for label in labels.tolist():
        tp = float(np.sum((y_true == label) & (y_pred == label)))
        fp = float(np.sum((y_true != label) & (y_pred == label)))
        fn = float(np.sum((y_true == label) & (y_pred != label)))
        support = int(np.sum(y_true == label))
        pred_support = int(np.sum(y_pred == label))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)

        rows.append(
            {
                "label_id": int(label),
                "label": (id2label or {}).get(int(label), str(label)),
                "support": support,
                "pred_support": pred_support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "train_support": int((train_support or {}).get(int(label), 0)),
            }
        )

    return rows


def build_head_tail_summary(per_class_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not per_class_rows:
        return {
            "strategy": "train_support_tertiles",
            "num_labels": 0,
            "groups": {},
        }

    sorted_rows = sorted(
        per_class_rows,
        key=lambda row: (-int(row.get("train_support", 0)), int(row.get("label_id", 0))),
    )

    total = len(sorted_rows)
    head_end = max(1, math.ceil(total / 3))
    mid_end = max(head_end + 1, math.ceil(2 * total / 3)) if total >= 3 else total

    groups = {
        "head": sorted_rows[:head_end],
        "mid": sorted_rows[head_end:mid_end],
        "tail": sorted_rows[mid_end:],
    }

    summary_groups: dict[str, Any] = {}
    for name, rows in groups.items():
        if not rows:
            summary_groups[name] = {
                "num_labels": 0,
                "avg_f1": 0.0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "total_support": 0,
                "total_train_support": 0,
                "labels": [],
            }
            continue

        summary_groups[name] = {
            "num_labels": len(rows),
            "avg_f1": float(np.mean([float(row["f1"]) for row in rows])),
            "avg_precision": float(np.mean([float(row["precision"]) for row in rows])),
            "avg_recall": float(np.mean([float(row["recall"]) for row in rows])),
            "total_support": int(sum(int(row["support"]) for row in rows)),
            "total_train_support": int(sum(int(row["train_support"]) for row in rows)),
            "labels": [str(row["label"]) for row in rows],
        }

    worst_rows = sorted(per_class_rows, key=lambda row: (float(row["f1"]), int(row["train_support"])))
    best_rows = sorted(per_class_rows, key=lambda row: (-float(row["f1"]), -int(row["train_support"])))

    return {
        "strategy": "train_support_tertiles",
        "num_labels": total,
        "groups": summary_groups,
        "worst_labels": worst_rows[:10],
        "best_labels": best_rows[:10],
    }


def compute_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict[str, Any]:
    """Compatibility helper for legacy scripts that expect sklearn report format."""
    labels = list(range(len(target_names)))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report: dict[str, Any] = {}
    supports = []
    f1_values = []

    for idx, name in zip(labels, target_names):
        tp = float(np.sum((y_true == idx) & (y_pred == idx)))
        fp = float(np.sum((y_true != idx) & (y_pred == idx)))
        fn = float(np.sum((y_true == idx) & (y_pred != idx)))
        support = int(np.sum(y_true == idx))
        supports.append(support)

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        f1_values.append(f1)

        report[name] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support,
        }

    metrics = compute_classification_metrics(y_true, y_pred)
    support_total = int(np.sum(supports))
    if support_total > 0:
        weighted_precision = float(
            np.sum([report[name]["precision"] * report[name]["support"] for name in target_names]) / support_total
        )
        weighted_recall = float(
            np.sum([report[name]["recall"] * report[name]["support"] for name in target_names]) / support_total
        )
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0

    report["accuracy"] = metrics["accuracy"]
    report["macro avg"] = {
        "precision": float(np.mean([report[name]["precision"] for name in target_names])) if target_names else 0.0,
        "recall": float(np.mean([report[name]["recall"] for name in target_names])) if target_names else 0.0,
        "f1-score": metrics["f1_macro"],
        "support": support_total,
    }
    report["weighted avg"] = {
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1-score": metrics["f1_weighted"],
        "support": support_total,
    }
    return report
