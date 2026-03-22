#!/usr/bin/env python
"""训练深度层次分类：3 类粗分类 + 粗类内细分类 + 验证集门控回退。"""

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

from charge_prediction.data_utils import read_jsonl
from charge_prediction.deep_models import (
    DeepChargeTrainer,
    DeepTrainingConfig,
    build_dataloaders,
    build_predict_dataloader,
    resolve_device,
    set_seed,
)
from charge_prediction.metrics import compute_classification_metrics


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
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
    )
    parser.add_argument("--fallback-to-flat", action="store_true")
    parser.add_argument("--coarse-threshold", type=float, default=-1.0)
    parser.add_argument("--coarse-margin-threshold", type=float, default=-1.0)
    return parser.parse_args()


def load_split(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def fit_label_encoder(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    uniq = sorted(set(labels))
    label2id = {x: i for i, x in enumerate(uniq)}
    id2label = {i: x for x, i in label2id.items()}
    return label2id, id2label


def encode_labels(series: pd.Series, label2id: dict[str, int]) -> np.ndarray:
    return series.map(label2id).fillna(-1).astype(int).to_numpy()


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


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


def build_config(args: argparse.Namespace) -> DeepTrainingConfig:
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
        selection_metric=args.selection_metric,
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
) -> tuple[DeepChargeTrainer, dict[str, float], dict[str, float], Path]:
    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        train_texts=train_texts,
        train_labels=train_labels,
        valid_texts=valid_texts,
        valid_labels=valid_labels,
        test_texts=test_texts,
        test_labels=test_labels,
        config=config,
    )
    trainer = DeepChargeTrainer(model_type, len(np.unique(train_labels)), config, device)
    best_valid, best_path = trainer.fit(train_loader, valid_loader, output_dir)
    test_metrics = trainer.evaluate(test_loader)
    return trainer, best_valid, test_metrics, best_path


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df = load_split(args.data_dir / "train_50k.jsonl")
    valid_df = load_split(args.data_dir / "valid_50k.jsonl")
    test_df = load_split(args.data_dir / "test_50k.jsonl")

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

    config = build_config(args)

    fine_train_loader, fine_valid_loader, fine_test_loader, _ = build_dataloaders(
        train_texts=x_train,
        train_labels=y_train_f,
        valid_texts=x_valid,
        valid_labels=y_valid_f,
        test_texts=x_test,
        test_labels=y_test_f,
        config=config,
    )
    coarse_train_loader, coarse_valid_loader, coarse_test_loader, _ = build_dataloaders(
        train_texts=x_train,
        train_labels=y_train_c,
        valid_texts=x_valid,
        valid_labels=y_valid_c,
        test_texts=x_test,
        test_labels=y_test_c,
        config=config,
    )

    flat_output_dir = args.output_dir / "flat_model"
    flat_trainer = DeepChargeTrainer(args.fine_model_type, len(fine_l2i), config, device)
    import torch

    fine_ckpt_path = Path(args.fine_checkpoint).expanduser() if args.fine_checkpoint else None
    if fine_ckpt_path is not None and fine_ckpt_path.exists() and fine_ckpt_path.is_file():
        flat_trainer.model.load_state_dict(torch.load(fine_ckpt_path, map_location=device))
        flat_best_valid = flat_trainer.evaluate(fine_valid_loader)
        flat_best_path = fine_ckpt_path
    else:
        flat_best_valid, flat_best_path = flat_trainer.fit(fine_train_loader, fine_valid_loader, flat_output_dir)
    flat_test_metrics = flat_trainer.evaluate(fine_test_loader)

    coarse_output_dir = args.output_dir / "coarse_model"
    coarse_trainer = DeepChargeTrainer(args.coarse_model_type, len(coarse_l2i), config, device)
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

        trainer, best_valid, test_metrics, best_path = train_subset_model(
            output_dir=args.output_dir / "local_models" / coarse_label,
            model_type=args.fine_model_type,
            config=config,
            device=device,
            train_texts=train_subset["fact"].tolist(),
            train_labels=encode_labels(train_subset["fine_label"], local_l2i),
            valid_texts=valid_subset["fact"].tolist(),
            valid_labels=encode_labels(valid_subset["fine_label"], local_l2i),
            test_texts=test_subset["fact"].tolist(),
            test_labels=encode_labels(test_subset["fine_label"], local_l2i),
        )
        local_models[int(coarse_id)] = {
            "coarse_label": coarse_label,
            "trainer": trainer,
            "best_valid": best_valid,
            "test_metrics": test_metrics,
            "checkpoint": str(best_path),
            "local_label2id": local_l2i,
            "local_id2global_fine_id": local_i2g,
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
        config=config,
    )
    test_routed_pred = routed_predictions(
        texts=x_test,
        coarse_logits=test_coarse_logits,
        flat_pred=test_flat_pred,
        local_models=local_models,
        config=config,
    )

    routing_config, valid_hier_pred, valid_hier_metrics, valid_routing_stats = tune_routing(
        y_true=y_valid_f,
        flat_pred=valid_flat_pred,
        routed_pred=valid_routed_pred,
        coarse_logits=valid_coarse_logits,
        metric_name=args.selection_metric,
        fallback_to_flat=args.fallback_to_flat,
        preset_confidence_threshold=args.coarse_threshold,
        preset_margin_threshold=args.coarse_margin_threshold,
    )

    test_probs = softmax(test_coarse_logits)
    test_coarse_conf, test_coarse_margin = top1_and_margin(test_probs)
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

    intermediate_rows = [
        {
            "split": "valid",
            "stage": "coarse",
            "accuracy": float(compute_classification_metrics(y_valid_c, valid_coarse_pred)["accuracy"]),
        },
        {"split": "valid", "stage": "fine_flat", "accuracy": float(valid_flat_metrics["accuracy"])},
        {"split": "valid", "stage": "fine_hier", "accuracy": float(valid_hier_metrics["accuracy"])},
        {
            "split": "test",
            "stage": "coarse",
            "accuracy": float(compute_classification_metrics(y_test_c, test_coarse_pred)["accuracy"]),
        },
        {"split": "test", "stage": "fine_flat", "accuracy": float(test_flat_metrics["accuracy"])},
        {"split": "test", "stage": "fine_hier", "accuracy": float(test_hier_metrics["accuracy"])},
    ]

    metrics = {
        "artifact_type": "deep_hierarchical",
        "task_mode": args.task_mode,
        "config": {
            "fine_model_type": args.fine_model_type,
            "coarse_model_type": args.coarse_model_type,
            "pretrained_model": args.pretrained_model,
            "device": str(device),
            "selection_metric": args.selection_metric,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "seed": args.seed,
        },
        "valid": {
            "coarse": compute_classification_metrics(y_valid_c, valid_coarse_pred),
            "fine_flat": valid_flat_metrics,
            "fine_hier": valid_hier_metrics,
        },
        "test": {
            "coarse": compute_classification_metrics(y_test_c, test_coarse_pred),
            "fine_flat": test_flat_metrics,
            "fine_hier": test_hier_metrics,
        },
        "routing_config": routing_config,
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
    (args.output_dir / "fine_label2id.json").write_text(json.dumps(fine_l2i, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "coarse_label2id.json").write_text(json.dumps(coarse_l2i, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] deep hierarchical finished")
    print("[Routing config]", routing_config)
    print("[Test coarse   ]", metrics["test"]["coarse"])
    print("[Test fine-flat]", metrics["test"]["fine_flat"])
    print("[Test fine-hier]", metrics["test"]["fine_hier"])


if __name__ == "__main__":
    main()
