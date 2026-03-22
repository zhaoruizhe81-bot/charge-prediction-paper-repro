"""Utilities for hierarchical fusion based on model score constraints."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import compute_classification_metrics


def _extract_classes(estimator: Any) -> np.ndarray:
    if hasattr(estimator, "classes_"):
        return np.asarray(estimator.classes_, dtype=int)

    if hasattr(estimator, "named_steps"):
        clf = estimator.named_steps.get("clf")
        if clf is not None and hasattr(clf, "classes_"):
            return np.asarray(clf.classes_, dtype=int)

    raise ValueError("Cannot infer estimator classes_.")


def score_matrix_from_estimator(
    estimator: Any,
    texts: list[str],
    num_classes: int,
) -> np.ndarray:
    """Return class-wise score matrix aligned to [0, num_classes)."""
    classes = _extract_classes(estimator)

    if hasattr(estimator, "predict_proba"):
        raw_scores = np.asarray(estimator.predict_proba(texts), dtype=float)
    elif hasattr(estimator, "decision_function"):
        raw_scores = np.asarray(estimator.decision_function(texts), dtype=float)
        if raw_scores.ndim == 1:
            if len(classes) != 2:
                raise ValueError("Binary decision_function shape mismatch with classes_.")
            raw_scores = np.stack([-raw_scores, raw_scores], axis=1)
    else:
        preds = np.asarray(estimator.predict(texts), dtype=int)
        raw_scores = np.zeros((len(texts), len(classes)), dtype=float)
        class_to_col = {int(label): idx for idx, label in enumerate(classes.tolist())}
        for row_idx, label in enumerate(preds.tolist()):
            col = class_to_col.get(int(label))
            if col is not None:
                raw_scores[row_idx, col] = 1.0

    if raw_scores.ndim != 2:
        raise ValueError("Expected score matrix with 2 dimensions.")

    aligned = np.full((raw_scores.shape[0], num_classes), -1e12, dtype=float)
    for col_idx, class_id in enumerate(classes.tolist()):
        aligned[:, int(class_id)] = raw_scores[:, col_idx]
    return aligned


def compute_margin(scores: np.ndarray) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D array.")
    if scores.shape[1] <= 1:
        return np.full(scores.shape[0], np.inf, dtype=float)
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def hierarchical_constrained_decode(
    fine_scores: np.ndarray,
    coarse_scores: np.ndarray,
    coarse_to_fine: dict[int, set[int]],
    top_k_coarse: int = 1,
    confidence_threshold: float | None = None,
    max_fine_margin: float | None = None,
) -> np.ndarray:
    """Decode with coarse-label constraints over fine-label scores."""
    if fine_scores.ndim != 2:
        raise ValueError("fine_scores must be a 2D array.")
    if coarse_scores.ndim != 2:
        raise ValueError("coarse_scores must be a 2D array.")
    if fine_scores.shape[0] != coarse_scores.shape[0]:
        raise ValueError("fine_scores and coarse_scores must share the same number of rows.")

    fine_pred = np.argmax(fine_scores, axis=1)
    if top_k_coarse <= 0 or coarse_scores.shape[1] == 0:
        return fine_pred

    ranked_coarse = np.argsort(coarse_scores, axis=1)[:, ::-1]
    margins = compute_margin(coarse_scores)
    fine_margins = compute_margin(fine_scores)
    threshold = float("-inf") if confidence_threshold is None else float(confidence_threshold)
    fine_threshold = float("inf") if max_fine_margin is None else float(max_fine_margin)
    max_top_k = min(top_k_coarse, coarse_scores.shape[1])
    num_fine_labels = fine_scores.shape[1]

    for idx in range(fine_scores.shape[0]):
        if margins[idx] < threshold:
            continue
        if fine_margins[idx] > fine_threshold:
            continue

        allowed: set[int] = set()
        for coarse_id in ranked_coarse[idx, :max_top_k].tolist():
            allowed.update(coarse_to_fine.get(int(coarse_id), set()))

        if not allowed:
            continue

        masked_scores = fine_scores[idx].copy()
        mask = np.ones(num_fine_labels, dtype=bool)
        mask[list(allowed)] = False
        masked_scores[mask] = -1e12
        fine_pred[idx] = int(np.argmax(masked_scores))

    return fine_pred


def _threshold_candidates(margins: np.ndarray) -> list[float]:
    if margins.size == 0:
        return [0.0]
    quantiles = np.quantile(margins, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    min_margin = float(np.min(margins) - 1e-8)
    max_margin = float(np.max(margins) + 1e-8)
    return sorted({float(value) for value in [min_margin, *quantiles, max_margin]})


def _is_better(candidate: dict[str, float], best: dict[str, float], metric_name: str) -> bool:
    candidate_metric = float(candidate.get(metric_name, 0.0))
    best_metric = float(best.get(metric_name, 0.0))

    if candidate_metric > best_metric + 1e-12:
        return True
    if abs(candidate_metric - best_metric) > 1e-12:
        return False

    candidate_acc = float(candidate.get("accuracy", 0.0))
    best_acc = float(best.get("accuracy", 0.0))
    return candidate_acc > best_acc + 1e-12


def tune_hierarchical_fusion(
    y_true: np.ndarray,
    fine_scores: np.ndarray,
    coarse_scores: np.ndarray,
    coarse_to_fine: dict[int, set[int]],
    metric_name: str = "f1_macro",
    max_top_k_coarse: int = 2,
) -> dict[str, Any]:
    """Tune hierarchical-fusion hyperparameters on a validation set."""
    valid_metric_name = metric_name if metric_name in {"accuracy", "f1_macro", "f1_micro", "f1_weighted"} else "f1_macro"
    base_pred = np.argmax(fine_scores, axis=1)
    base_metrics = compute_classification_metrics(y_true, base_pred)

    best_pred = np.array(base_pred, copy=True)
    best_metrics = dict(base_metrics)
    best_config = {
        "use_hier_fusion": False,
        "top_k_coarse": 0,
        "confidence_threshold": None,
        "max_fine_margin": None,
    }

    if coarse_scores.shape[1] == 0 or max_top_k_coarse <= 0:
        return {
            "metric_name": valid_metric_name,
            "base_metrics": base_metrics,
            "best_metrics": best_metrics,
            "best_pred": best_pred,
            "best_config": best_config,
        }

    margins = compute_margin(coarse_scores)
    fine_margins = compute_margin(fine_scores)
    thresholds = _threshold_candidates(margins)
    fine_thresholds = _threshold_candidates(fine_margins)
    max_top_k = min(int(max_top_k_coarse), coarse_scores.shape[1])

    for top_k in range(1, max_top_k + 1):
        for threshold in thresholds:
            for fine_threshold in fine_thresholds:
                fused_pred = hierarchical_constrained_decode(
                    fine_scores=fine_scores,
                    coarse_scores=coarse_scores,
                    coarse_to_fine=coarse_to_fine,
                    top_k_coarse=top_k,
                    confidence_threshold=threshold,
                    max_fine_margin=fine_threshold,
                )
                fused_metrics = compute_classification_metrics(y_true, fused_pred)
                if _is_better(fused_metrics, best_metrics, valid_metric_name):
                    best_pred = fused_pred
                    best_metrics = fused_metrics
                    best_config = {
                        "use_hier_fusion": True,
                        "top_k_coarse": int(top_k),
                        "confidence_threshold": float(threshold),
                        "max_fine_margin": float(fine_threshold),
                    }

    return {
        "metric_name": valid_metric_name,
        "base_metrics": base_metrics,
        "best_metrics": best_metrics,
        "best_pred": best_pred,
        "best_config": best_config,
    }


def metric_delta(
    improved: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    keys = ["accuracy", "f1_macro", "f1_micro", "f1_weighted"]
    return {key: float(improved.get(key, 0.0) - baseline.get(key, 0.0)) for key in keys}
