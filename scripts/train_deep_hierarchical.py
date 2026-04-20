#!/usr/bin/env python
"""训练深度层次分类：3 类粗分类 + 粗类内细分类 + 验证集门控回退。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.data_utils import read_jsonl
from charge_prediction.deep_models import (
    DeepChargeTrainer,
    DeepTrainingConfig,
    build_dataloaders,
    build_predict_dataloader,
    build_tokenizer_cache_key,
    compute_class_weights,
    compute_sample_weights,
    resolve_device,
    set_seed,
)
from charge_prediction.metrics import build_head_tail_summary, compute_classification_metrics, compute_per_class_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep hierarchical charge classifier")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_110_paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper/deep_hierarchical_fc"))
    parser.add_argument("--task-mode", type=str, default="single_label_110", choices=["single_label_110"])
    parser.add_argument("--pretrained-model", type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--fine-model-type", type=str, default="fc", choices=["fc", "rcnn"])
    parser.add_argument("--coarse-model-type", type=str, default="fc", choices=["fc", "rcnn"])
    parser.add_argument("--fine-checkpoint", type=str, default="")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--rcnn-hidden-size", type=int, default=256)
    parser.add_argument("--rcnn-num-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "recall_macro", "recall_micro", "recall_weighted", "f1_macro", "f1_micro", "f1_weighted", "f1_score"],
    )
    parser.add_argument("--fallback-to-flat", action="store_true")
    parser.add_argument("--coarse-threshold", type=float, default=-1.0)
    parser.add_argument("--coarse-margin-threshold", type=float, default=-1.0)
    parser.add_argument("--local-init-from-flat", action="store_true")
    parser.add_argument(
        "--hier-fusion-mode",
        type=str,
        default="flat_local_coarse",
        choices=["routing", "flat_local_coarse"],
    )
    parser.add_argument(
        "--coarse-fusion-weights",
        type=float,
        nargs="*",
        default=[0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
    )
    parser.add_argument(
        "--local-fusion-weights",
        type=float,
        nargs="*",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--optimize-profile",
        type=str,
        default="baseline",
        choices=["baseline", "windows_4060ti_best"],
    )
    parser.add_argument("--loss", type=str, default="", choices=["", "ce", "weighted_ce", "focal"])
    parser.add_argument("--label-smoothing", type=float, default=-1.0)
    parser.add_argument("--sampler", type=str, default="", choices=["", "none", "weighted"])
    parser.add_argument("--pin-memory", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--persistent-workers", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--prefetch-factor", type=int, default=0)
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def limit_df(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


def fit_label_encoder(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    uniq = sorted(set(labels))
    label2id = {x: i for i, x in enumerate(uniq)}
    id2label = {i: x for x, i in label2id.items()}
    return label2id, id2label


def encode_labels(series: pd.Series, label2id: dict[str, int]) -> np.ndarray:
    return series.map(label2id).fillna(-1).astype(int).to_numpy()


def build_fine_to_coarse_mapping(train_df: pd.DataFrame) -> dict[str, str]:
    mapping_df = (
        train_df[["fine_label", "coarse_label"]]
        .drop_duplicates()
        .drop_duplicates(subset=["fine_label"], keep="first")
    )
    return mapping_df.set_index("fine_label")["coarse_label"].to_dict()


def normalize_coarse_labels(df: pd.DataFrame, fine_to_coarse: dict[str, str]) -> pd.DataFrame:
    result = df.copy()
    # Some processed valid/test rows carry coarse labels that are inconsistent with
    # the fine label. Normalize them to the train-derived mapping so local
    # hierarchical models only see labels that belong to their coarse bucket.
    result["coarse_label"] = result["fine_label"].map(fine_to_coarse).fillna(result["coarse_label"])
    return result


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def log_softmax(x: np.ndarray) -> np.ndarray:
    probabilities = softmax(x)
    return np.log(probabilities.clip(min=1e-12))


def top1_and_margin(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    confidence = np.max(probabilities, axis=1)
    if probabilities.shape[1] <= 1:
        return confidence, np.full(confidence.shape[0], np.inf, dtype=float)
    sorted_probs = np.sort(probabilities, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    return confidence, margin


def threshold_candidates(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return [0.0]
    quantiles = np.quantile(values, [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]).tolist()
    return sorted({round(float(v), 6) for v in [0.0, *quantiles, 1.0]})


def is_better(candidate: dict[str, float], best: dict[str, float], metric_name: str) -> bool:
    candidate_metric = float(candidate.get(metric_name, 0.0))
    best_metric = float(best.get(metric_name, 0.0))
    if candidate_metric > best_metric + 1e-12:
        return True
    if abs(candidate_metric - best_metric) > 1e-12:
        return False
    return float(candidate.get("accuracy", 0.0)) > float(best.get("accuracy", 0.0)) + 1e-12


def resolve_toggle(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def apply_optimization_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.optimize_profile == "windows_4060ti_best":
        args.loss = "weighted_ce" if not args.loss else args.loss
        args.label_smoothing = 0.05 if args.label_smoothing < 0 else args.label_smoothing
        args.sampler = "weighted" if not args.sampler else args.sampler
        args.selection_metric = "f1_macro"
        args.early_stopping_patience = 3
        if args.pin_memory == "auto":
            args.pin_memory = "on"
        if args.persistent_workers == "auto":
            args.persistent_workers = "off"
        if args.prefetch_factor <= 0:
            args.prefetch_factor = 2

    if not args.loss:
        args.loss = "ce"
    if args.label_smoothing < 0:
        args.label_smoothing = 0.0
    if not args.sampler:
        args.sampler = "none"
    return args


def build_config(args: argparse.Namespace, *, is_coarse: bool) -> DeepTrainingConfig:
    return DeepTrainingConfig(
        pretrained_model_name=args.pretrained_model,
        max_length=args.max_length,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        dropout=args.dropout,
        rcnn_hidden_size=args.rcnn_hidden_size,
        rcnn_num_layers=args.rcnn_num_layers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        num_workers=args.num_workers,
        selection_metric="accuracy" if is_coarse else args.selection_metric,
        optimize_profile=args.optimize_profile,
        loss_name="ce" if is_coarse else args.loss,
        label_smoothing=0.0 if is_coarse else args.label_smoothing,
        sampler_name="none" if is_coarse else args.sampler,
        pin_memory=resolve_toggle(args.pin_memory),
        persistent_workers=resolve_toggle(args.persistent_workers),
        prefetch_factor=args.prefetch_factor if args.prefetch_factor > 0 else None,
        enable_tokenizer_cache=args.optimize_profile != "baseline",
    )


def train_subset_model(
    output_dir: Path,
    model_type: str,
    config: DeepTrainingConfig,
    device,
    train_texts: list[str],
    train_labels: np.ndarray,
    valid_texts: list[str],
    valid_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    class_weights: np.ndarray | None = None,
    train_sample_weights: np.ndarray | None = None,
    train_sampler_name: str | None = None,
    train_cache_key: str | None = None,
    valid_cache_key: str | None = None,
    test_cache_key: str | None = None,
    flat_state_dict: dict[str, torch.Tensor] | None = None,
    local_id2global_fine_id: dict[int, int] | None = None,
) -> tuple[DeepChargeTrainer, dict[str, float], dict[str, float], Path]:
    train_mask = train_labels >= 0
    valid_mask = valid_labels >= 0
    test_mask = test_labels >= 0

    train_texts = [text for text, keep in zip(train_texts, train_mask.tolist()) if keep]
    valid_texts = [text for text, keep in zip(valid_texts, valid_mask.tolist()) if keep]
    test_texts = [text for text, keep in zip(test_texts, test_mask.tolist()) if keep]

    train_labels = train_labels[train_mask]
    valid_labels = valid_labels[valid_mask]
    test_labels = test_labels[test_mask]
    if train_sample_weights is not None:
        train_sample_weights = train_sample_weights[train_mask]

    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        train_texts=train_texts,
        train_labels=train_labels,
        valid_texts=valid_texts,
        valid_labels=valid_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        config=config,
        train_cache_key=train_cache_key,
        valid_cache_key=valid_cache_key,
        test_cache_key=test_cache_key,
        train_sample_weights=train_sample_weights,
        train_sampler_name=train_sampler_name,
    )
    num_labels = int(np.max(train_labels)) + 1 if train_labels.size else 1
    trainer = DeepChargeTrainer(model_type, num_labels, config, device, class_weights=class_weights)
    if flat_state_dict is not None and local_id2global_fine_id is not None:
        initialize_local_model_from_flat(trainer, flat_state_dict, local_id2global_fine_id)
    best_valid, best_path = trainer.fit(train_loader, valid_loader, output_dir)
    test_metrics = trainer.evaluate(test_loader)
    return trainer, best_valid, test_metrics, best_path


def initialize_local_model_from_flat(
    trainer: DeepChargeTrainer,
    flat_state_dict: dict[str, torch.Tensor],
    local_id2global_fine_id: dict[int, int],
) -> None:
    local_state = trainer.model.state_dict()
    updated_state: dict[str, torch.Tensor] = {}
    copied = 0
    for key, local_value in local_state.items():
        if key == "classifier.weight" and key in flat_state_dict:
            source = flat_state_dict[key]
            if source.ndim == 2 and source.shape[1] == local_value.shape[1]:
                target = local_value.detach().clone()
                for local_id, global_id in local_id2global_fine_id.items():
                    if int(global_id) < source.shape[0] and int(local_id) < target.shape[0]:
                        target[int(local_id)] = source[int(global_id)].detach().to(target.device)
                updated_state[key] = target
                copied += 1
                continue
        if key == "classifier.bias" and key in flat_state_dict:
            source = flat_state_dict[key]
            if source.ndim == 1:
                target = local_value.detach().clone()
                for local_id, global_id in local_id2global_fine_id.items():
                    if int(global_id) < source.shape[0] and int(local_id) < target.shape[0]:
                        target[int(local_id)] = source[int(global_id)].detach().to(target.device)
                updated_state[key] = target
                copied += 1
                continue
        source = flat_state_dict.get(key)
        if source is not None and tuple(source.shape) == tuple(local_value.shape):
            updated_state[key] = source.detach().to(local_value.device)
            copied += 1
        else:
            updated_state[key] = local_value
    trainer.model.load_state_dict(updated_state)
    print(f"[Local init] copied {copied} tensors from flat checkpoint")


def routed_predictions(
    texts: list[str],
    coarse_logits: np.ndarray,
    flat_pred: np.ndarray,
    local_models: dict[int, dict[str, object]],
    config: DeepTrainingConfig,
) -> np.ndarray:
    routed = np.array(flat_pred, copy=True)
    if coarse_logits.shape[0] == 0:
        return routed

    coarse_pred = np.argmax(coarse_logits, axis=1)
    for coarse_id, model_info in local_models.items():
        indices = np.where(coarse_pred == coarse_id)[0]
        if len(indices) == 0:
            continue
        subset_texts = [texts[idx] for idx in indices.tolist()]
        loader = build_predict_dataloader(subset_texts, config=config)
        local_trainer = model_info["trainer"]
        local_logits = local_trainer.collect_logits(loader)
        local_pred = np.argmax(local_logits, axis=1)
        local_to_global = model_info["local_id2global_fine_id"]
        routed[indices] = np.asarray([local_to_global[int(item)] for item in local_pred], dtype=int)
    return routed


def collect_local_global_log_probs(
    texts: list[str],
    local_models: dict[int, dict[str, object]],
    config: DeepTrainingConfig,
    num_global_labels: int,
) -> np.ndarray:
    result = np.full((len(texts), num_global_labels), -30.0, dtype=np.float32)
    if not texts:
        return result
    for model_info in local_models.values():
        trainer = model_info["trainer"]
        local_logits = trainer.collect_logits(build_predict_dataloader(texts, config=config))
        local_log_probs = log_softmax(local_logits)
        local_to_global = model_info["local_id2global_fine_id"]
        for local_id, global_id in local_to_global.items():
            result[:, int(global_id)] = local_log_probs[:, int(local_id)]
    return result


def fine_to_coarse_array(
    num_fine_labels: int,
    local_models: dict[int, dict[str, object]],
) -> np.ndarray:
    mapping = np.full(num_fine_labels, -1, dtype=int)
    for coarse_id, model_info in local_models.items():
        for global_id in model_info["local_id2global_fine_id"].values():
            mapping[int(global_id)] = int(coarse_id)
    return mapping


def coarse_log_probs_for_fine_labels(coarse_logits: np.ndarray, fine_to_coarse: np.ndarray) -> np.ndarray:
    coarse_log_probs = log_softmax(coarse_logits)
    result = np.zeros((coarse_logits.shape[0], len(fine_to_coarse)), dtype=np.float32)
    for fine_id, coarse_id in enumerate(fine_to_coarse.tolist()):
        if coarse_id >= 0:
            result[:, fine_id] = coarse_log_probs[:, coarse_id]
    return result


def tune_flat_local_coarse_fusion(
    y_true: np.ndarray,
    flat_logits: np.ndarray,
    coarse_logits: np.ndarray,
    local_log_probs: np.ndarray,
    fine_to_coarse: np.ndarray,
    *,
    metric_name: str,
    coarse_weights: list[float],
    local_weights: list[float],
) -> tuple[dict[str, object], np.ndarray, dict[str, float]]:
    flat_log_probs = log_softmax(flat_logits)
    coarse_fine_log_probs = coarse_log_probs_for_fine_labels(coarse_logits, fine_to_coarse)
    best_pred = np.argmax(flat_log_probs, axis=1)
    best_metrics = compute_classification_metrics(y_true, best_pred)
    best_config: dict[str, object] = {
        "use_hierarchical_routing": False,
        "fusion_mode": "flat_baseline",
        "coarse_weight": 0.0,
        "local_weight": 0.0,
    }

    for coarse_weight in sorted(set(float(item) for item in coarse_weights)):
        for local_weight in sorted(set(float(item) for item in local_weights)):
            if coarse_weight == 0.0 and local_weight == 0.0:
                continue
            scores = flat_log_probs + coarse_weight * coarse_fine_log_probs + local_weight * local_log_probs
            pred = np.argmax(scores, axis=1)
            metrics = compute_classification_metrics(y_true, pred)
            if is_better(metrics, best_metrics, metric_name):
                best_pred = pred
                best_metrics = metrics
                best_config = {
                    "use_hierarchical_routing": True,
                    "fusion_mode": "flat_local_coarse",
                    "coarse_weight": float(coarse_weight),
                    "local_weight": float(local_weight),
                }
    return best_config, best_pred, best_metrics


def apply_flat_local_coarse_fusion(
    flat_logits: np.ndarray,
    coarse_logits: np.ndarray,
    local_log_probs: np.ndarray,
    fine_to_coarse: np.ndarray,
    config: dict[str, object],
) -> tuple[np.ndarray, dict[str, int]]:
    if not bool(config.get("use_hierarchical_routing", False)):
        pred = np.argmax(flat_logits, axis=1)
        return pred, {"num_routed": 0, "num_fallback": int(len(pred))}
    scores = (
        log_softmax(flat_logits)
        + float(config.get("coarse_weight", 0.0)) * coarse_log_probs_for_fine_labels(coarse_logits, fine_to_coarse)
        + float(config.get("local_weight", 0.0)) * local_log_probs
    )
    pred = np.argmax(scores, axis=1)
    return pred, {"num_routed": int(len(pred)), "num_fallback": 0}


def apply_gating(
    flat_pred: np.ndarray,
    routed_pred: np.ndarray,
    coarse_confidence: np.ndarray,
    coarse_margin: np.ndarray,
    confidence_threshold: float,
    margin_threshold: float,
    fallback_to_flat: bool,
) -> tuple[np.ndarray, dict[str, int]]:
    if not fallback_to_flat:
        return routed_pred, {"num_routed": int(len(routed_pred)), "num_fallback": 0}

    use_routing = (coarse_confidence >= confidence_threshold) & (coarse_margin >= margin_threshold)
    pred = np.where(use_routing, routed_pred, flat_pred)
    return pred, {
        "num_routed": int(np.sum(use_routing)),
        "num_fallback": int(len(pred) - np.sum(use_routing)),
    }


def tune_routing(
    y_true: np.ndarray,
    flat_pred: np.ndarray,
    routed_pred: np.ndarray,
    coarse_logits: np.ndarray,
    metric_name: str,
    fallback_to_flat: bool,
    preset_confidence_threshold: float,
    preset_margin_threshold: float,
) -> tuple[dict[str, object], np.ndarray, dict[str, float], dict[str, int]]:
    probs = softmax(coarse_logits)
    coarse_confidence, coarse_margin = top1_and_margin(probs)

    thresholds = [preset_confidence_threshold] if preset_confidence_threshold >= 0 else threshold_candidates(coarse_confidence)
    margins = [preset_margin_threshold] if preset_margin_threshold >= 0 else threshold_candidates(coarse_margin)

    best_pred = np.array(flat_pred, copy=True)
    best_metrics = compute_classification_metrics(y_true, best_pred)
    best_config: dict[str, object] = {
        "use_hierarchical_routing": False,
        "fallback_to_flat": fallback_to_flat,
        "coarse_threshold": None,
        "coarse_margin_threshold": None,
    }
    best_stats = {"num_routed": 0, "num_fallback": int(len(best_pred))}

    for threshold in thresholds:
        for margin_threshold in margins:
            pred, stats = apply_gating(
                flat_pred=flat_pred,
                routed_pred=routed_pred,
                coarse_confidence=coarse_confidence,
                coarse_margin=coarse_margin,
                confidence_threshold=float(threshold),
                margin_threshold=float(margin_threshold),
                fallback_to_flat=fallback_to_flat,
            )
            metrics = compute_classification_metrics(y_true, pred)
            if is_better(metrics, best_metrics, metric_name):
                best_pred = pred
                best_metrics = metrics
                best_config = {
                    "use_hierarchical_routing": True,
                    "fallback_to_flat": fallback_to_flat,
                    "coarse_threshold": float(threshold),
                    "coarse_margin_threshold": float(margin_threshold),
                }
                best_stats = stats

    return best_config, best_pred, best_metrics, best_stats


def export_hierarchical_diagnostics(
    output_dir: Path,
    *,
    id2label: dict[int, str],
    train_labels: np.ndarray,
    y_true: np.ndarray,
    variants: dict[str, np.ndarray],
) -> None:
    label_ids = sorted(id2label.keys())
    train_support = {int(label_id): int(count) for label_id, count in enumerate(np.bincount(train_labels, minlength=len(label_ids)))}

    all_rows: list[dict[str, object]] = []
    head_tail_summary: dict[str, object] = {}
    for variant_name, y_pred in variants.items():
        rows = compute_per_class_metrics(
            y_true,
            y_pred,
            label_ids=label_ids,
            id2label=id2label,
            train_support=train_support,
        )
        for row in rows:
            row["variant"] = variant_name
        all_rows.extend(rows)
        head_tail_summary[variant_name] = build_head_tail_summary(rows)

    pd.DataFrame(all_rows).to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    (output_dir / "head_tail_metrics.json").write_text(
        json.dumps(head_tail_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_routing_note(routing_config: dict[str, object]) -> str:
    if not bool(routing_config.get("use_hierarchical_routing", False)):
        return "Hierarchical training completed, but validation tuning kept the flat baseline as the final routing strategy."
    return "Hierarchical routing was enabled by validation tuning."


def main() -> None:
    args = apply_optimization_profile(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_path = args.data_dir / "train_50k.jsonl"
    valid_path = args.data_dir / "valid_50k.jsonl"
    test_path = args.data_dir / "test_50k.jsonl"

    train_df = load_split(train_path)
    valid_df = load_split(valid_path)
    test_df = load_split(test_path)

    fine_to_coarse = build_fine_to_coarse_mapping(train_df)
    train_df = normalize_coarse_labels(train_df, fine_to_coarse)
    valid_df = normalize_coarse_labels(valid_df, fine_to_coarse)
    test_df = normalize_coarse_labels(test_df, fine_to_coarse)

    train_df = limit_df(train_df, args.max_train_samples, args.seed)
    train_fine_labels = set(train_df["fine_label"].tolist())
    train_coarse_labels = set(train_df["coarse_label"].tolist())
    valid_df = valid_df[
        valid_df["fine_label"].isin(train_fine_labels) & valid_df["coarse_label"].isin(train_coarse_labels)
    ].reset_index(drop=True)
    test_df = test_df[
        test_df["fine_label"].isin(train_fine_labels) & test_df["coarse_label"].isin(train_coarse_labels)
    ].reset_index(drop=True)
    valid_df = limit_df(valid_df, args.max_valid_samples, args.seed)
    test_df = limit_df(test_df, args.max_test_samples, args.seed)
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Smoke/limited splits must all be non-empty after label filtering.")

    fine_l2i, fine_i2l = fit_label_encoder(train_df["fine_label"].tolist())
    coarse_l2i, coarse_i2l = fit_label_encoder(train_df["coarse_label"].tolist())

    x_train = train_df["fact"].tolist()
    x_valid = valid_df["fact"].tolist()
    x_test = test_df["fact"].tolist()

    y_train_f = encode_labels(train_df["fine_label"], fine_l2i)
    y_valid_f = encode_labels(valid_df["fine_label"], fine_l2i)
    y_test_f = encode_labels(test_df["fine_label"], fine_l2i)
    y_train_c = encode_labels(train_df["coarse_label"], coarse_l2i)
    y_valid_c = encode_labels(valid_df["coarse_label"], coarse_l2i)
    y_test_c = encode_labels(test_df["coarse_label"], coarse_l2i)

    fine_config = build_config(args, is_coarse=False)
    coarse_config = build_config(args, is_coarse=True)

    fine_class_weights = compute_class_weights(y_train_f, num_labels=len(fine_l2i))
    fine_sample_weights = compute_sample_weights(y_train_f, fine_class_weights) if args.sampler == "weighted" else None

    fine_train_cache_key = build_tokenizer_cache_key(
        train_path,
        fine_config,
        extra=f"hier-fine-train|seed={args.seed}",
    )
    fine_valid_cache_key = build_tokenizer_cache_key(
        valid_path,
        fine_config,
        extra="hier-fine-valid",
    )
    fine_test_cache_key = build_tokenizer_cache_key(
        test_path,
        fine_config,
        extra="hier-fine-test",
    )
    coarse_train_cache_key = build_tokenizer_cache_key(
        train_path,
        coarse_config,
        extra=f"hier-coarse-train|seed={args.seed}",
    )
    coarse_valid_cache_key = build_tokenizer_cache_key(
        valid_path,
        coarse_config,
        extra="hier-coarse-valid",
    )
    coarse_test_cache_key = build_tokenizer_cache_key(
        test_path,
        coarse_config,
        extra="hier-coarse-test",
    )

    fine_train_loader, fine_valid_loader, fine_test_loader, _ = build_dataloaders(
        train_texts=x_train,
        train_labels=y_train_f,
        valid_texts=x_valid,
        valid_labels=y_valid_f,
        test_texts=x_test,
        test_labels=y_test_f,
        config=fine_config,
        train_cache_key=fine_train_cache_key,
        valid_cache_key=fine_valid_cache_key,
        test_cache_key=fine_test_cache_key,
        train_sample_weights=fine_sample_weights,
        train_sampler_name=args.sampler,
    )
    coarse_train_loader, coarse_valid_loader, coarse_test_loader, _ = build_dataloaders(
        train_texts=x_train,
        train_labels=y_train_c,
        valid_texts=x_valid,
        valid_labels=y_valid_c,
        test_texts=x_test,
        test_labels=y_test_c,
        config=coarse_config,
        train_cache_key=coarse_train_cache_key,
        valid_cache_key=coarse_valid_cache_key,
        test_cache_key=coarse_test_cache_key,
        train_sampler_name="none",
    )

    flat_output_dir = args.output_dir / "flat_model"
    flat_trainer = DeepChargeTrainer(
        args.fine_model_type,
        len(fine_l2i),
        fine_config,
        device,
        class_weights=fine_class_weights if args.loss in {"weighted_ce", "focal"} else None,
    )
    fine_ckpt_path = Path(args.fine_checkpoint).expanduser() if args.fine_checkpoint else None
    if fine_ckpt_path is not None and fine_ckpt_path.exists() and fine_ckpt_path.is_file():
        flat_trainer.model.load_state_dict(torch.load(fine_ckpt_path, map_location=device))
        flat_best_valid = flat_trainer.evaluate(fine_valid_loader)
        flat_best_path = fine_ckpt_path
    else:
        flat_best_valid, flat_best_path = flat_trainer.fit(fine_train_loader, fine_valid_loader, flat_output_dir)
    flat_test_metrics = flat_trainer.evaluate(fine_test_loader)
    flat_state_dict = torch.load(flat_best_path, map_location=device) if args.local_init_from_flat else None

    coarse_output_dir = args.output_dir / "coarse_model"
    coarse_trainer = DeepChargeTrainer(args.coarse_model_type, len(coarse_l2i), coarse_config, device)
    coarse_best_valid, coarse_best_path = coarse_trainer.fit(coarse_train_loader, coarse_valid_loader, coarse_output_dir)
    coarse_test_metrics = coarse_trainer.evaluate(coarse_test_loader)

    local_models: dict[int, dict[str, object]] = {}
    for coarse_label, coarse_id in coarse_l2i.items():
        train_subset = train_df[train_df["coarse_label"] == coarse_label].reset_index(drop=True)
        valid_subset = valid_df[valid_df["coarse_label"] == coarse_label].reset_index(drop=True)
        test_subset = test_df[test_df["coarse_label"] == coarse_label].reset_index(drop=True)

        local_fine_labels = sorted(train_subset["fine_label"].unique().tolist())
        local_l2i = {label: idx for idx, label in enumerate(local_fine_labels)}
        local_i2g = {idx: fine_l2i[label] for label, idx in local_l2i.items()}
        local_train_labels = encode_labels(train_subset["fine_label"], local_l2i)
        local_valid_labels = encode_labels(valid_subset["fine_label"], local_l2i)
        local_test_labels = encode_labels(test_subset["fine_label"], local_l2i)
        local_class_weights = compute_class_weights(local_train_labels, num_labels=len(local_l2i))
        local_sample_weights = compute_sample_weights(local_train_labels, local_class_weights) if args.sampler == "weighted" else None

        trainer, best_valid, test_metrics, best_path = train_subset_model(
            output_dir=args.output_dir / "local_models" / coarse_label,
            model_type=args.fine_model_type,
            config=fine_config,
            device=device,
            train_texts=train_subset["fact"].tolist(),
            train_labels=local_train_labels,
            valid_texts=valid_subset["fact"].tolist(),
            valid_labels=local_valid_labels,
            test_texts=test_subset["fact"].tolist(),
            test_labels=local_test_labels,
            class_weights=local_class_weights if args.loss in {"weighted_ce", "focal"} else None,
            train_sample_weights=local_sample_weights,
            train_sampler_name=args.sampler,
            train_cache_key=build_tokenizer_cache_key(train_path, fine_config, extra=f"local-train|{coarse_label}|{len(train_subset)}"),
            valid_cache_key=build_tokenizer_cache_key(valid_path, fine_config, extra=f"local-valid|{coarse_label}|{len(valid_subset)}"),
            test_cache_key=build_tokenizer_cache_key(test_path, fine_config, extra=f"local-test|{coarse_label}|{len(test_subset)}"),
            flat_state_dict=flat_state_dict,
            local_id2global_fine_id=local_i2g if args.local_init_from_flat else None,
        )
        local_models[int(coarse_id)] = {
            "coarse_label": coarse_label,
            "trainer": trainer,
            "best_valid": best_valid,
            "test_metrics": test_metrics,
            "checkpoint": str(best_path),
            "local_label2id": local_l2i,
            "local_id2global_fine_id": local_i2g,
            "num_train": int(len(train_subset)),
            "num_valid": int(len(valid_subset)),
            "num_test": int(len(test_subset)),
        }

    valid_flat_logits = flat_trainer.collect_logits(fine_valid_loader)
    test_flat_logits = flat_trainer.collect_logits(fine_test_loader)
    valid_coarse_logits = coarse_trainer.collect_logits(coarse_valid_loader)
    test_coarse_logits = coarse_trainer.collect_logits(coarse_test_loader)

    valid_flat_pred = np.argmax(valid_flat_logits, axis=1)
    test_flat_pred = np.argmax(test_flat_logits, axis=1)
    valid_coarse_pred = np.argmax(valid_coarse_logits, axis=1)
    test_coarse_pred = np.argmax(test_coarse_logits, axis=1)

    valid_routed_pred = routed_predictions(
        texts=x_valid,
        coarse_logits=valid_coarse_logits,
        flat_pred=valid_flat_pred,
        local_models=local_models,
        config=fine_config,
    )
    test_routed_pred = routed_predictions(
        texts=x_test,
        coarse_logits=test_coarse_logits,
        flat_pred=test_flat_pred,
        local_models=local_models,
        config=fine_config,
    )

    if args.hier_fusion_mode == "flat_local_coarse":
        fine_to_coarse = fine_to_coarse_array(len(fine_l2i), local_models)
        valid_local_log_probs = collect_local_global_log_probs(
            x_valid,
            local_models,
            fine_config,
            len(fine_l2i),
        )
        test_local_log_probs = collect_local_global_log_probs(
            x_test,
            local_models,
            fine_config,
            len(fine_l2i),
        )
        routing_config, valid_hier_pred, valid_hier_metrics = tune_flat_local_coarse_fusion(
            y_true=y_valid_f,
            flat_logits=valid_flat_logits,
            coarse_logits=valid_coarse_logits,
            local_log_probs=valid_local_log_probs,
            fine_to_coarse=fine_to_coarse,
            metric_name=fine_config.selection_metric,
            coarse_weights=args.coarse_fusion_weights,
            local_weights=args.local_fusion_weights,
        )
        test_hier_pred, test_routing_stats = apply_flat_local_coarse_fusion(
            test_flat_logits,
            test_coarse_logits,
            test_local_log_probs,
            fine_to_coarse,
            routing_config,
        )
        valid_routing_stats = {
            "num_routed": int(len(valid_hier_pred)) if bool(routing_config.get("use_hierarchical_routing", False)) else 0,
            "num_fallback": int(0 if bool(routing_config.get("use_hierarchical_routing", False)) else len(valid_hier_pred)),
        }
    else:
        routing_config, valid_hier_pred, valid_hier_metrics, valid_routing_stats = tune_routing(
            y_true=y_valid_f,
            flat_pred=valid_flat_pred,
            routed_pred=valid_routed_pred,
            coarse_logits=valid_coarse_logits,
            metric_name=fine_config.selection_metric,
            fallback_to_flat=args.fallback_to_flat,
            preset_confidence_threshold=args.coarse_threshold,
            preset_margin_threshold=args.coarse_margin_threshold,
        )
        test_probs = softmax(test_coarse_logits)
        test_coarse_conf, test_coarse_margin = top1_and_margin(test_probs)
        if not bool(routing_config.get("use_hierarchical_routing", False)):
            test_hier_pred = np.array(test_flat_pred, copy=True)
            test_routing_stats = {"num_routed": 0, "num_fallback": int(len(test_flat_pred))}
        else:
            test_hier_pred, test_routing_stats = apply_gating(
                flat_pred=test_flat_pred,
                routed_pred=test_routed_pred,
                coarse_confidence=test_coarse_conf,
                coarse_margin=test_coarse_margin,
                confidence_threshold=float(routing_config.get("coarse_threshold") or 0.0),
                margin_threshold=float(routing_config.get("coarse_margin_threshold") or 0.0),
                fallback_to_flat=bool(routing_config.get("fallback_to_flat", False)),
            )

    valid_flat_metrics = compute_classification_metrics(y_valid_f, valid_flat_pred)
    test_flat_metrics = compute_classification_metrics(y_test_f, test_flat_pred)
    test_hier_metrics = compute_classification_metrics(y_test_f, test_hier_pred)
    routing_note = build_routing_note(routing_config)

    def intermediate_row(split: str, stage: str, metric_values: dict[str, float]) -> dict[str, object]:
        return {
            "split": split,
            "stage": stage,
            "accuracy": float(metric_values.get("accuracy", 0.0)),
            "recall_macro": float(metric_values.get("recall_macro", 0.0)),
            "recall_micro": float(metric_values.get("recall_micro", 0.0)),
            "f1_score": float(metric_values.get("f1_score", 0.0)),
            "f1_macro": float(metric_values.get("f1_macro", 0.0)),
            "f1_micro": float(metric_values.get("f1_micro", 0.0)),
        }

    valid_coarse_metrics = compute_classification_metrics(y_valid_c, valid_coarse_pred)
    test_coarse_metrics = compute_classification_metrics(y_test_c, test_coarse_pred)
    intermediate_rows = [
        intermediate_row("valid", "coarse", valid_coarse_metrics),
        intermediate_row("valid", "fine_flat", valid_flat_metrics),
        intermediate_row("valid", "fine_hier", valid_hier_metrics),
        intermediate_row("test", "coarse", test_coarse_metrics),
        intermediate_row("test", "fine_flat", test_flat_metrics),
        intermediate_row("test", "fine_hier", test_hier_metrics),
    ]

    metrics = {
        "artifact_type": "deep_hierarchical",
        "task_mode": args.task_mode,
        "config": {
            "fine_model_type": args.fine_model_type,
            "coarse_model_type": args.coarse_model_type,
            "pretrained_model": args.pretrained_model,
            "device": str(device),
            "selection_metric": fine_config.selection_metric,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
            "max_train_samples": args.max_train_samples,
            "max_valid_samples": args.max_valid_samples,
            "max_test_samples": args.max_test_samples,
            "local_init_from_flat": args.local_init_from_flat,
            "hier_fusion_mode": args.hier_fusion_mode,
            "coarse_fusion_weights": args.coarse_fusion_weights,
            "local_fusion_weights": args.local_fusion_weights,
            "optimize_profile": args.optimize_profile,
            "loss": args.loss,
            "label_smoothing": args.label_smoothing,
            "sampler": args.sampler,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.persistent_workers,
            "prefetch_factor": args.prefetch_factor,
        },
        "valid": {
            "coarse": valid_coarse_metrics,
            "fine_flat": valid_flat_metrics,
            "fine_hier": valid_hier_metrics,
        },
        "test": {
            "coarse": test_coarse_metrics,
            "fine_flat": test_flat_metrics,
            "fine_hier": test_hier_metrics,
        },
        "routing_config": routing_config,
        "routing_note": routing_note,
        "routing_stats": {
            "valid": valid_routing_stats,
            "test": test_routing_stats,
        },
        "intermediate_rows": intermediate_rows,
        "flat_best_valid": flat_best_valid,
        "flat_test": flat_test_metrics,
        "coarse_best_valid": coarse_best_valid,
        "coarse_test": coarse_test_metrics,
        "flat_checkpoint": str(flat_best_path),
        "coarse_checkpoint": str(coarse_best_path),
        "num_fine_labels": len(fine_l2i),
        "num_coarse_labels": len(coarse_l2i),
    }

    model_bundle = {
        "artifact_type": "deep_hierarchical",
        "task_mode": args.task_mode,
        "fine_model_type": args.fine_model_type,
        "coarse_model_type": args.coarse_model_type,
        "pretrained_model": args.pretrained_model,
        "max_length": args.max_length,
        "eval_batch_size": args.eval_batch_size,
        "num_workers": args.num_workers,
        "dropout": args.dropout,
        "rcnn_hidden_size": args.rcnn_hidden_size,
        "rcnn_num_layers": args.rcnn_num_layers,
        "routing_config": routing_config,
        "fine_label2id": fine_l2i,
        "fine_id2label": {str(k): v for k, v in fine_i2l.items()},
        "coarse_label2id": coarse_l2i,
        "coarse_id2label": {str(k): v for k, v in coarse_i2l.items()},
        "flat_checkpoint": str(flat_best_path),
        "coarse_checkpoint": str(coarse_best_path),
        "local_models": {
            str(coarse_id): {
                "coarse_label": model_info["coarse_label"],
                "checkpoint": model_info["checkpoint"],
                "local_label2id": model_info["local_label2id"],
                "local_id2global_fine_id": {str(k): int(v) for k, v in model_info["local_id2global_fine_id"].items()},
            }
            for coarse_id, model_info in local_models.items()
        },
    }

    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "model_bundle.json").write_text(
        json.dumps(model_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "eval_outputs.npz",
        y_valid=y_valid_f.astype(np.int64),
        valid_logits=valid_flat_logits.astype(np.float32),
        valid_pred=valid_hier_pred.astype(np.int64),
        y_test=y_test_f.astype(np.int64),
        test_logits=test_flat_logits.astype(np.float32),
        test_pred=test_hier_pred.astype(np.int64),
        valid_coarse_logits=valid_coarse_logits.astype(np.float32),
        test_coarse_logits=test_coarse_logits.astype(np.float32),
    )
    (args.output_dir / "fine_label2id.json").write_text(json.dumps(fine_l2i, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "coarse_label2id.json").write_text(json.dumps(coarse_l2i, ensure_ascii=False, indent=2), encoding="utf-8")
    export_hierarchical_diagnostics(
        args.output_dir,
        id2label=fine_i2l,
        train_labels=y_train_f,
        y_true=y_test_f,
        variants={
            "fine_flat": test_flat_pred,
            "fine_hier": test_hier_pred,
        },
    )
    routing_diagnostics = {
        "routing_config": routing_config,
        "routing_note": routing_note,
        "routing_stats": {
            "valid": valid_routing_stats,
            "test": test_routing_stats,
        },
        "local_models": {
            str(coarse_id): {
                "coarse_label": model_info["coarse_label"],
                "num_local_labels": len(model_info["local_label2id"]),
                "num_train": model_info["num_train"],
                "num_valid": model_info["num_valid"],
                "num_test": model_info["num_test"],
                "best_valid": model_info["best_valid"],
                "test_metrics": model_info["test_metrics"],
                "checkpoint": model_info["checkpoint"],
            }
            for coarse_id, model_info in local_models.items()
        },
    }
    (args.output_dir / "routing_diagnostics.json").write_text(
        json.dumps(routing_diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[Done] deep hierarchical finished")
    print("[Routing config]", routing_config)
    print("[Routing note  ]", routing_note)
    print("[Test coarse   ]", metrics["test"]["coarse"])
    print("[Test fine-flat]", metrics["test"]["fine_flat"])
    print("[Test fine-hier]", metrics["test"]["fine_hier"])


if __name__ == "__main__":
    main()
