#!/usr/bin/env python
"""Train shared-encoder hierarchical charge classifiers and validation-tuned ensembles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.data_utils import read_jsonl
from charge_prediction.deep_models import (
    DeepTrainingConfig,
    HierarchicalDeepTrainer,
    build_hierarchical_dataloaders,
    build_tokenizer_cache_key,
    compute_class_weights,
    compute_sample_weights,
    resolve_device,
    set_seed,
)
from charge_prediction.metrics import compute_classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multitask hierarchical charge models")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_110_paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper/charge_hier_multitask"))
    parser.add_argument("--flat-dir", type=Path, default=Path(""))
    parser.add_argument("--models", nargs="+", default=["rcnn"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 3407])
    parser.add_argument("--pretrained-model", type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--rcnn-hidden-size", type=int, default=256)
    parser.add_argument("--rcnn-num-layers", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--selection-metric", type=str, default="accuracy")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--loss", type=str, default="weighted_ce", choices=["ce", "weighted_ce", "focal"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--sampler", type=str, default="none", choices=["none", "weighted"])
    parser.add_argument("--class-weight-max", type=float, default=3.0)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.3)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.2)
    parser.add_argument(
        "--coarse-fusion-weights",
        type=float,
        nargs="*",
        default=[0.0, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
    )
    parser.add_argument("--max-ensemble-members", type=int, default=6)
    parser.add_argument(
        "--optimize-profile",
        type=str,
        default="baseline",
        choices=["baseline", "windows_4060ti_best"],
    )
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
    label2id = {label: index for index, label in enumerate(uniq)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def build_fine_to_coarse_mapping(train_df: pd.DataFrame) -> dict[str, str]:
    mapping_df = (
        train_df[["fine_label", "coarse_label"]]
        .drop_duplicates()
        .drop_duplicates(subset=["fine_label"], keep="first")
    )
    return mapping_df.set_index("fine_label")["coarse_label"].to_dict()


def normalize_coarse_labels(df: pd.DataFrame, fine_to_coarse: dict[str, str]) -> pd.DataFrame:
    result = df.copy()
    result["coarse_label"] = result["fine_label"].map(fine_to_coarse).fillna(result["coarse_label"])
    return result


def encode_labels(series: pd.Series, label2id: dict[str, int]) -> np.ndarray:
    return series.map(label2id).fillna(-1).astype(int).to_numpy()


def resolve_toggle(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def build_config(args: argparse.Namespace, seed: int) -> DeepTrainingConfig:
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
        seed=seed,
        num_workers=args.num_workers,
        selection_metric=args.selection_metric,
        optimize_profile=args.optimize_profile,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        sampler_name=args.sampler,
        pin_memory=resolve_toggle(args.pin_memory),
        persistent_workers=resolve_toggle(args.persistent_workers),
        prefetch_factor=args.prefetch_factor if args.prefetch_factor > 0 else None,
        enable_tokenizer_cache=args.optimize_profile != "baseline",
    )


def clipped_class_weights(labels: np.ndarray, num_labels: int, max_weight: float) -> np.ndarray:
    weights = compute_class_weights(labels, num_labels=num_labels)
    if max_weight > 0:
        weights = np.clip(weights, 0.0, float(max_weight))
        weights = weights / max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    logsum = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True).clip(min=1e-12))
    return shifted - logsum


def coarse_log_probs_for_fine(coarse_logits: np.ndarray, fine_to_coarse_ids: np.ndarray) -> np.ndarray:
    coarse_log_probs = log_softmax(coarse_logits)
    result = np.zeros((coarse_logits.shape[0], len(fine_to_coarse_ids)), dtype=np.float32)
    for fine_id, coarse_id in enumerate(fine_to_coarse_ids.tolist()):
        if coarse_id >= 0:
            result[:, fine_id] = coarse_log_probs[:, int(coarse_id)]
    return result


def fused_scores(fine_logits: np.ndarray, coarse_logits: np.ndarray, fine_to_coarse_ids: np.ndarray, coarse_weight: float) -> np.ndarray:
    return log_softmax(fine_logits) + float(coarse_weight) * coarse_log_probs_for_fine(coarse_logits, fine_to_coarse_ids)


def metric_score(metrics: dict[str, float], objective: str) -> tuple[float, ...]:
    accuracy = float(metrics.get("accuracy", 0.0))
    recall_macro = float(metrics.get("recall_macro", 0.0))
    recall_micro = float(metrics.get("recall_micro", 0.0))
    f1_score = float(metrics.get("f1_score", 0.0))
    if objective == "recall":
        return (recall_macro + recall_micro, accuracy, f1_score)
    if objective == "balanced":
        return (accuracy + f1_score + 0.5 * recall_macro + 0.25 * recall_micro, accuracy, f1_score)
    return (accuracy, f1_score, recall_macro, recall_micro)


def evaluate_scores(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = np.argmax(scores, axis=1)
    return {key: float(value) for key, value in compute_classification_metrics(y_true, pred).items()}


def select_candidate(candidates: list[dict[str, Any]], y_valid: np.ndarray, objective: str) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No hierarchical candidates were produced.")
    best: dict[str, Any] | None = None
    best_score: tuple[float, ...] | None = None
    for candidate in candidates:
        metrics = evaluate_scores(y_valid, candidate["valid_scores"])
        score = metric_score(metrics, objective)
        if best_score is None or score > best_score:
            best = dict(candidate)
            best["valid_metrics"] = metrics
            best_score = score
    if best is None:
        raise RuntimeError("No candidate selected.")
    return best


def load_flat_candidates(flat_dir: Path, y_valid: np.ndarray, y_test: np.ndarray) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not flat_dir or str(flat_dir) == "." or not flat_dir.exists():
        return candidates
    for model_dir in sorted(flat_dir.glob("*")):
        path = model_dir / "eval_outputs.npz"
        if not path.exists():
            continue
        with np.load(path) as data:
            if "valid_logits" not in data or "test_logits" not in data:
                continue
            if not (np.array_equal(data["y_valid"], y_valid) and np.array_equal(data["y_test"], y_test)):
                continue
            candidates.append(
                {
                    "name": f"flat_{model_dir.name}",
                    "members": [f"flat:{model_dir.name}"],
                    "valid_scores": log_softmax(data["valid_logits"]),
                    "test_scores": log_softmax(data["test_logits"]),
                    "uses_multitask": False,
                }
            )
    return candidates


def build_ensemble_candidates(pool: list[dict[str, Any]], y_valid: np.ndarray, max_members: int) -> list[dict[str, Any]]:
    if len(pool) < 2:
        return []
    ranked = sorted(pool, key=lambda item: metric_score(evaluate_scores(y_valid, item["valid_scores"]), "accuracy"), reverse=True)
    candidates: list[dict[str, Any]] = []
    upper = min(max(2, int(max_members)), len(ranked))
    for size in range(2, upper + 1):
        members = ranked[:size]
        if not any(item.get("uses_multitask", False) for item in members):
            continue
        candidates.append(
            {
                "name": f"ensemble_top{size}",
                "members": [member["name"] for member in members],
                "valid_scores": np.mean([member["valid_scores"] for member in members], axis=0),
                "test_scores": np.mean([member["test_scores"] for member in members], axis=0),
                "uses_multitask": True,
            }
        )
    return candidates


def add_intermediate_row(rows: list[dict[str, Any]], split: str, stage: str, metrics: dict[str, float], *, model: str, source: str) -> None:
    rows.append(
        {
            "split": split,
            "stage": stage,
            "model": model,
            "source": source,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "recall_macro": float(metrics.get("recall_macro", 0.0)),
            "recall_micro": float(metrics.get("recall_micro", 0.0)),
            "f1_score": float(metrics.get("f1_score", 0.0)),
        }
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train_50k.jsonl"
    valid_path = args.data_dir / "valid_50k.jsonl"
    test_path = args.data_dir / "test_50k.jsonl"

    train_df = limit_df(load_split(train_path), args.max_train_samples, args.seeds[0])
    valid_df = limit_df(load_split(valid_path), args.max_valid_samples, args.seeds[0])
    test_df = limit_df(load_split(test_path), args.max_test_samples, args.seeds[0])

    fine_to_coarse = build_fine_to_coarse_mapping(train_df)
    train_df = normalize_coarse_labels(train_df, fine_to_coarse)
    valid_df = normalize_coarse_labels(valid_df, fine_to_coarse)
    test_df = normalize_coarse_labels(test_df, fine_to_coarse)

    fine_l2i, fine_i2l = fit_label_encoder(train_df["fine_label"].tolist())
    coarse_l2i, coarse_i2l = fit_label_encoder(train_df["coarse_label"].tolist())
    train_fine_set = set(fine_l2i)
    train_coarse_set = set(coarse_l2i)

    valid_df = valid_df[valid_df["fine_label"].isin(train_fine_set) & valid_df["coarse_label"].isin(train_coarse_set)].reset_index(drop=True)
    test_df = test_df[test_df["fine_label"].isin(train_fine_set) & test_df["coarse_label"].isin(train_coarse_set)].reset_index(drop=True)

    y_train_f = encode_labels(train_df["fine_label"], fine_l2i)
    y_train_c = encode_labels(train_df["coarse_label"], coarse_l2i)
    y_valid_f = encode_labels(valid_df["fine_label"], fine_l2i)
    y_valid_c = encode_labels(valid_df["coarse_label"], coarse_l2i)
    y_test_f = encode_labels(test_df["fine_label"], fine_l2i)
    y_test_c = encode_labels(test_df["coarse_label"], coarse_l2i)

    x_train = train_df["fact"].tolist()
    x_valid = valid_df["fact"].tolist()
    x_test = test_df["fact"].tolist()

    fine_to_coarse_ids = np.full((len(fine_l2i),), -1, dtype=np.int64)
    for fine_label, fine_id in fine_l2i.items():
        fine_to_coarse_ids[fine_id] = coarse_l2i[fine_to_coarse[fine_label]]

    device = resolve_device(args.device)
    run_metrics: dict[str, Any] = {}
    pool_candidates: list[dict[str, Any]] = []
    hier_candidates: list[dict[str, Any]] = []
    intermediate_rows: list[dict[str, Any]] = []

    for model_name in args.models:
        for seed in args.seeds:
            run_name = f"{model_name}_seed{seed}"
            run_dir = args.output_dir / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            set_seed(seed)
            config = build_config(args, seed)
            fine_weights = clipped_class_weights(y_train_f, len(fine_l2i), args.class_weight_max)
            coarse_weights = clipped_class_weights(y_train_c, len(coarse_l2i), args.class_weight_max)
            train_sample_weights = compute_sample_weights(y_train_f, fine_weights) if args.sampler == "weighted" else None

            train_loader, valid_loader, test_loader, _ = build_hierarchical_dataloaders(
                train_texts=x_train,
                train_fine_labels=y_train_f,
                train_coarse_labels=y_train_c,
                valid_texts=x_valid,
                valid_fine_labels=y_valid_f,
                valid_coarse_labels=y_valid_c,
                test_texts=x_test,
                test_fine_labels=y_test_f,
                test_coarse_labels=y_test_c,
                config=config,
                train_cache_key=build_tokenizer_cache_key(train_path, config, extra=f"mt-train|{run_name}|{len(x_train)}"),
                valid_cache_key=build_tokenizer_cache_key(valid_path, config, extra=f"mt-valid|{run_name}|{len(x_valid)}"),
                test_cache_key=build_tokenizer_cache_key(test_path, config, extra=f"mt-test|{run_name}|{len(x_test)}"),
                train_sample_weights=train_sample_weights,
                train_sampler_name=args.sampler,
            )

            trainer = HierarchicalDeepTrainer(
                model_type=model_name,
                num_fine_labels=len(fine_l2i),
                num_coarse_labels=len(coarse_l2i),
                config=config,
                device=device,
                fine_to_coarse_ids=fine_to_coarse_ids,
                fine_class_weights=fine_weights if args.loss in {"weighted_ce", "focal"} else None,
                coarse_class_weights=coarse_weights,
                coarse_loss_weight=args.coarse_loss_weight,
                consistency_loss_weight=args.consistency_loss_weight,
            )
            best_valid, best_path = trainer.fit(train_loader, valid_loader, run_dir)
            test_eval = trainer.evaluate(test_loader)
            valid_fine_logits, valid_coarse_logits = trainer.collect_logits(valid_loader)
            test_fine_logits, test_coarse_logits = trainer.collect_logits(test_loader)
            valid_pred = np.argmax(valid_fine_logits, axis=1)
            test_pred = np.argmax(test_fine_logits, axis=1)

            np.savez_compressed(
                run_dir / "eval_outputs.npz",
                y_valid=y_valid_f.astype(np.int64),
                valid_logits=valid_fine_logits.astype(np.float32),
                valid_pred=valid_pred.astype(np.int64),
                y_test=y_test_f.astype(np.int64),
                test_logits=test_fine_logits.astype(np.float32),
                test_pred=test_pred.astype(np.int64),
                y_valid_coarse=y_valid_c.astype(np.int64),
                valid_coarse_logits=valid_coarse_logits.astype(np.float32),
                y_test_coarse=y_test_c.astype(np.int64),
                test_coarse_logits=test_coarse_logits.astype(np.float32),
            )

            run_metrics[run_name] = {
                "model": model_name,
                "seed": seed,
                "best_valid": best_valid,
                "test": test_eval,
                "checkpoint": str(best_path),
            }
            add_intermediate_row(intermediate_rows, "valid", "coarse", best_valid["coarse"], model=model_name, source=run_name)
            add_intermediate_row(intermediate_rows, "valid", "fine_multitask", best_valid["fine"], model=model_name, source=run_name)
            add_intermediate_row(intermediate_rows, "test", "coarse", test_eval["coarse"], model=model_name, source=run_name)
            add_intermediate_row(intermediate_rows, "test", "fine_multitask", test_eval["fine"], model=model_name, source=run_name)

            for coarse_weight in args.coarse_fusion_weights:
                candidate = {
                    "name": f"mt_{run_name}_cw{coarse_weight:g}",
                    "members": [run_name],
                    "valid_scores": fused_scores(valid_fine_logits, valid_coarse_logits, fine_to_coarse_ids, coarse_weight),
                    "test_scores": fused_scores(test_fine_logits, test_coarse_logits, fine_to_coarse_ids, coarse_weight),
                    "uses_multitask": True,
                    "coarse_weight": float(coarse_weight),
                }
                pool_candidates.append(candidate)
                hier_candidates.append(candidate)

    flat_candidates = load_flat_candidates(args.flat_dir, y_valid_f, y_test_f)
    pool_candidates.extend(flat_candidates)
    ensemble_candidates = build_ensemble_candidates(pool_candidates, y_valid_f, args.max_ensemble_members)
    hier_candidates.extend(ensemble_candidates)

    selected_accuracy = select_candidate(hier_candidates, y_valid_f, "accuracy")
    selected_recall = select_candidate(hier_candidates, y_valid_f, "recall")
    selected_balanced = select_candidate(hier_candidates, y_valid_f, "balanced")

    selections = {
        "fine_hier": selected_accuracy,
        "fine_hier_accuracy": selected_accuracy,
        "fine_hier_recall": selected_recall,
        "fine_hier_balanced": selected_balanced,
    }

    test_metrics: dict[str, dict[str, float]] = {}
    valid_metrics: dict[str, dict[str, float]] = {}
    routing_configs: dict[str, dict[str, Any]] = {}
    for stage, selected in selections.items():
        valid_pred = np.argmax(selected["valid_scores"], axis=1)
        test_pred = np.argmax(selected["test_scores"], axis=1)
        valid_metric = {key: float(value) for key, value in compute_classification_metrics(y_valid_f, valid_pred).items()}
        test_metric = {key: float(value) for key, value in compute_classification_metrics(y_test_f, test_pred).items()}
        valid_metrics[stage] = valid_metric
        test_metrics[stage] = test_metric
        routing_configs[stage] = {
            "candidate": selected["name"],
            "members": selected.get("members", []),
            "coarse_weight": selected.get("coarse_weight"),
        }
        add_intermediate_row(intermediate_rows, "valid", stage, valid_metric, model="multitask_ensemble", source=args.output_dir.name)
        add_intermediate_row(intermediate_rows, "test", stage, test_metric, model="multitask_ensemble", source=args.output_dir.name)

    test_metrics["coarse"] = max(
        (item["test"]["coarse"] for item in run_metrics.values()),
        key=lambda metrics: float(metrics.get("accuracy", 0.0)),
    )
    valid_metrics["coarse"] = max(
        (item["best_valid"]["coarse"] for item in run_metrics.values()),
        key=lambda metrics: float(metrics.get("accuracy", 0.0)),
    )

    accuracy_pred_valid = np.argmax(selected_accuracy["valid_scores"], axis=1)
    accuracy_pred_test = np.argmax(selected_accuracy["test_scores"], axis=1)
    np.savez_compressed(
        args.output_dir / "eval_outputs.npz",
        y_valid=y_valid_f.astype(np.int64),
        valid_logits=selected_accuracy["valid_scores"].astype(np.float32),
        valid_pred=accuracy_pred_valid.astype(np.int64),
        y_test=y_test_f.astype(np.int64),
        test_logits=selected_accuracy["test_scores"].astype(np.float32),
        test_pred=accuracy_pred_test.astype(np.int64),
    )

    candidate_rows = []
    for candidate in hier_candidates:
        valid_metric = evaluate_scores(y_valid_f, candidate["valid_scores"])
        candidate_rows.append(
            {
                "name": candidate["name"],
                "members": ",".join(candidate.get("members", [])),
                "valid_accuracy": valid_metric["accuracy"],
                "valid_recall_macro": valid_metric["recall_macro"],
                "valid_recall_micro": valid_metric["recall_micro"],
                "valid_f1_score": valid_metric["f1_score"],
            }
        )
    pd.DataFrame(candidate_rows).sort_values(by=["valid_accuracy", "valid_f1_score"], ascending=False).to_csv(
        args.output_dir / "ensemble_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = {
        "artifact_type": "charge_hier_multitask",
        "config": {
            "models": args.models,
            "seeds": args.seeds,
            "pretrained_model": args.pretrained_model,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "loss": args.loss,
            "label_smoothing": args.label_smoothing,
            "sampler": args.sampler,
            "class_weight_max": args.class_weight_max,
            "coarse_loss_weight": args.coarse_loss_weight,
            "consistency_loss_weight": args.consistency_loss_weight,
            "flat_dir": str(args.flat_dir),
        },
        "label2id": fine_l2i,
        "id2label": {str(k): v for k, v in fine_i2l.items()},
        "coarse_label2id": coarse_l2i,
        "coarse_id2label": {str(k): v for k, v in coarse_i2l.items()},
        "fine_to_coarse_ids": fine_to_coarse_ids.tolist(),
        "runs": run_metrics,
        "valid": valid_metrics,
        "test": test_metrics,
        "routing_configs": routing_configs,
        "intermediate_rows": intermediate_rows,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(intermediate_rows).to_csv(args.output_dir / "intermediate_rows.csv", index=False, encoding="utf-8-sig")

    print("[Done] charge multitask hierarchical training finished")
    print("[Selected accuracy]", routing_configs["fine_hier_accuracy"])
    print("[Test accuracy]", test_metrics["fine_hier_accuracy"])
    print("[Selected recall]", routing_configs["fine_hier_recall"])
    print("[Test recall]", test_metrics["fine_hier_recall"])


if __name__ == "__main__":
    main()
