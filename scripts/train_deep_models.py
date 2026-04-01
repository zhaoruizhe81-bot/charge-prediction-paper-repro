#!/usr/bin/env python
"""训练深度学习平层模型：BERT+FC 与 BERT+RCNN。"""

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
    build_tokenizer_cache_key,
    compute_class_weights,
    compute_sample_weights,
    resolve_device,
    set_seed,
)
from charge_prediction.metrics import build_head_tail_summary, compute_per_class_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train flat BERT models for charge prediction")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_110_paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper/deep_models"))
    parser.add_argument("--models", nargs="+", default=["fc", "rcnn"])
    parser.add_argument("--pretrained-model", type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--task-mode", type=str, default="single_label_110", choices=["single_label_110"])
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "f1_macro", "f1_micro", "f1_weighted"],
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
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


def build_label_mapping(train_df: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(train_df["fine_label"].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def encode_labels(df: pd.DataFrame, label2id: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    encoded = df["fine_label"].map(label2id).fillna(-1).astype(int).to_numpy()
    mask = encoded >= 0
    return encoded[mask], mask


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


def build_config(args: argparse.Namespace) -> DeepTrainingConfig:
    enable_tokenizer_cache = args.optimize_profile != "baseline"
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
        optimize_profile=args.optimize_profile,
        loss_name=args.loss,
        label_smoothing=args.label_smoothing,
        sampler_name=args.sampler,
        pin_memory=resolve_toggle(args.pin_memory),
        persistent_workers=resolve_toggle(args.persistent_workers),
        prefetch_factor=args.prefetch_factor if args.prefetch_factor > 0 else None,
        enable_tokenizer_cache=enable_tokenizer_cache,
    )


def export_diagnostics(
    output_dir: Path,
    *,
    id2label: dict[int, str],
    train_labels: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    label_ids = sorted(id2label.keys())
    train_support = {int(label_id): int(count) for label_id, count in enumerate(np.bincount(train_labels, minlength=len(label_ids)))}
    rows = compute_per_class_metrics(
        y_true,
        y_pred,
        label_ids=label_ids,
        id2label=id2label,
        train_support=train_support,
    )
    pd.DataFrame(rows).to_csv(output_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    summary = build_head_tail_summary(rows)
    (output_dir / "head_tail_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


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

    train_df = limit_df(train_df, args.max_train_samples, args.seed)
    valid_df = limit_df(valid_df, args.max_valid_samples, args.seed)
    test_df = limit_df(test_df, args.max_test_samples, args.seed)

    label2id, id2label = build_label_mapping(train_df)
    y_train, train_mask = encode_labels(train_df, label2id)
    y_valid, valid_mask = encode_labels(valid_df, label2id)
    y_test, test_mask = encode_labels(test_df, label2id)

    x_train = train_df.loc[train_mask, "fact"].tolist()
    x_valid = valid_df.loc[valid_mask, "fact"].tolist()
    x_test = test_df.loc[test_mask, "fact"].tolist()

    config = build_config(args)

    class_weights = compute_class_weights(y_train, num_labels=len(label2id))
    train_sample_weights = compute_sample_weights(y_train, class_weights) if args.sampler == "weighted" else None

    train_cache_key = build_tokenizer_cache_key(
        train_path,
        config,
        extra=f"split=train|max={args.max_train_samples}|seed={args.seed}",
    )
    valid_cache_key = build_tokenizer_cache_key(
        valid_path,
        config,
        extra=f"split=valid|max={args.max_valid_samples}|seed={args.seed}",
    )
    test_cache_key = build_tokenizer_cache_key(
        test_path,
        config,
        extra=f"split=test|max={args.max_test_samples}|seed={args.seed}",
    )

    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        train_texts=x_train,
        train_labels=y_train,
        valid_texts=x_valid,
        valid_labels=y_valid,
        test_texts=x_test,
        test_labels=y_test,
        config=config,
        train_cache_key=train_cache_key,
        valid_cache_key=valid_cache_key,
        test_cache_key=test_cache_key,
        train_sample_weights=train_sample_weights,
        train_sampler_name=args.sampler,
    )

    all_metrics: dict[str, dict[str, object]] = {}

    for model_name in args.models:
        model_output_dir = args.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        trainer = DeepChargeTrainer(
            model_type=model_name,
            num_labels=len(label2id),
            config=config,
            device=device,
            class_weights=class_weights if args.loss in {"weighted_ce", "focal"} else None,
        )

        print(f"\n[Training] {model_name} on {device}")
        best_valid_metrics, best_model_path = trainer.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            output_dir=model_output_dir,
        )
        test_metrics = trainer.evaluate(test_loader)
        test_logits = trainer.collect_logits(test_loader)
        test_preds = np.argmax(test_logits, axis=1) if test_logits.size else np.empty((0,), dtype=int)
        export_diagnostics(
            model_output_dir,
            id2label=id2label,
            train_labels=y_train,
            y_true=y_test,
            y_pred=test_preds,
        )

        bundle = {
            "artifact_type": "deep_flat",
            "task_mode": args.task_mode,
            "model_type": model_name,
            "pretrained_model": args.pretrained_model,
            "checkpoint_path": str(best_model_path),
            "device": str(device),
            "config": {
                "max_length": args.max_length,
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "warmup_ratio": args.warmup_ratio,
                "early_stopping_patience": args.early_stopping_patience,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "rcnn_hidden_size": args.rcnn_hidden_size,
                "rcnn_num_layers": args.rcnn_num_layers,
                "selection_metric": args.selection_metric,
                "seed": args.seed,
                "optimize_profile": args.optimize_profile,
                "loss": args.loss,
                "label_smoothing": args.label_smoothing,
                "sampler": args.sampler,
                "pin_memory": args.pin_memory,
                "persistent_workers": args.persistent_workers,
                "prefetch_factor": args.prefetch_factor,
            },
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        }
        (model_output_dir / "model_bundle.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        all_metrics[model_name] = {
            "best_valid": best_valid_metrics,
            "test": test_metrics,
            "best_model_path": str(best_model_path),
            "model_bundle_path": str(model_output_dir / "model_bundle.json"),
            "config": bundle["config"],
        }

        print(f"[Best Valid] {best_valid_metrics}")
        print(f"[Test      ] {test_metrics}")

    (args.output_dir / "metrics.json").write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "label2id.json").write_text(
        json.dumps(label2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "id2label.json").write_text(
        json.dumps(id2label, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[Done] Deep flat models finished")


if __name__ == "__main__":
    main()
