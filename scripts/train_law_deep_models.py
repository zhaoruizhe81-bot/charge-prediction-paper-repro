#!/usr/bin/env python
"""Train flat BERT models for multi-label law-article recommendation."""

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

from charge_prediction.data_utils import build_multilabel_matrix, read_jsonl
from charge_prediction.deep_models import (
    DeepTrainingConfig,
    MultiLabelDeepTrainer,
    build_multilabel_dataloaders,
    build_tokenizer_cache_key,
    compute_multilabel_pos_weights,
    resolve_device,
    set_seed,
)
from charge_prediction.metrics import (
    compute_multilabel_metrics,
    compute_multilabel_per_label_metrics,
    multilabel_predictions_from_scores,
    tune_multilabel_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train law-article multi-label BERT models")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed_law"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper/law_deep"))
    parser.add_argument("--models", nargs="+", default=["fc", "rcnn"])
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="f1_score",
        choices=["accuracy", "recall_macro", "recall_micro", "f1_macro", "f1_micro", "f1_score"],
    )
    parser.add_argument("--thresholds", type=float, nargs="*", default=[0.2, 0.3, 0.4, 0.45, 0.5, 0.6])
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-valid-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--pin-memory", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--persistent-workers", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--prefetch-factor", type=int, default=0)
    parser.add_argument("--disable-tokenizer-cache", action="store_true")
    return parser.parse_args()


def resolve_toggle(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "on"


def load_split(path: Path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def limit_df(df: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples and len(df) > max_samples:
        return df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


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
        pin_memory=resolve_toggle(args.pin_memory),
        persistent_workers=resolve_toggle(args.persistent_workers),
        prefetch_factor=args.prefetch_factor if args.prefetch_factor > 0 else None,
        enable_tokenizer_cache=not args.disable_tokenizer_cache,
    )


def export_model_outputs(
    output_dir: Path,
    *,
    y_valid: np.ndarray,
    valid_scores: np.ndarray,
    valid_pred: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    test_pred: np.ndarray,
    id2label: dict[int, str],
) -> None:
    np.savez_compressed(
        output_dir / "eval_outputs.npz",
        y_valid=y_valid.astype(np.int8),
        valid_scores=valid_scores.astype(np.float32),
        valid_pred=valid_pred.astype(np.int8),
        y_test=y_test.astype(np.int8),
        test_scores=test_scores.astype(np.float32),
        test_pred=test_pred.astype(np.int8),
    )
    rows = compute_multilabel_per_label_metrics(y_test, test_pred, id2label=id2label)
    pd.DataFrame(rows).to_csv(output_dir / "per_label_metrics.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df = limit_df(load_split(args.data_dir / "train.jsonl"), args.max_train_samples, args.seed)
    valid_df = limit_df(load_split(args.data_dir / "valid.jsonl"), args.max_valid_samples, args.seed)
    test_df = limit_df(load_split(args.data_dir / "test.jsonl"), args.max_test_samples, args.seed)
    label2id = json.loads((args.data_dir / "label2id.json").read_text(encoding="utf-8"))
    id2label = {int(index): label for label, index in label2id.items()}

    y_train = build_multilabel_matrix(train_df["article_numbers"], label2id)
    y_valid = build_multilabel_matrix(valid_df["article_numbers"], label2id)
    y_test = build_multilabel_matrix(test_df["article_numbers"], label2id)
    x_train = train_df["fact"].tolist()
    x_valid = valid_df["fact"].tolist()
    x_test = test_df["fact"].tolist()

    config = build_config(args)
    pos_weight = compute_multilabel_pos_weights(y_train)

    train_loader, valid_loader, test_loader, _ = build_multilabel_dataloaders(
        train_texts=x_train,
        train_labels=y_train,
        valid_texts=x_valid,
        valid_labels=y_valid,
        test_texts=x_test,
        test_labels=y_test,
        config=config,
        train_cache_key=build_tokenizer_cache_key(args.data_dir / "train.jsonl", config, extra=f"law-train|{len(train_df)}"),
        valid_cache_key=build_tokenizer_cache_key(args.data_dir / "valid.jsonl", config, extra=f"law-valid|{len(valid_df)}"),
        test_cache_key=build_tokenizer_cache_key(args.data_dir / "test.jsonl", config, extra=f"law-test|{len(test_df)}"),
    )

    all_metrics: dict[str, dict[str, object]] = {}
    for model_name in args.models:
        model_dir = args.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        trainer = MultiLabelDeepTrainer(
            model_type=model_name,
            num_labels=len(label2id),
            config=config,
            device=device,
            pos_weight=pos_weight,
        )
        print(f"\n[Training law] {model_name} on {device}")
        best_valid, best_path = trainer.fit(train_loader, valid_loader, model_dir)
        valid_scores = 1.0 / (1.0 + np.exp(-trainer.collect_logits(valid_loader)))
        test_scores = 1.0 / (1.0 + np.exp(-trainer.collect_logits(test_loader)))
        threshold, tuned_valid = tune_multilabel_threshold(
            y_valid,
            valid_scores,
            metric_name=args.selection_metric,
            thresholds=args.thresholds,
        )
        valid_pred = multilabel_predictions_from_scores(valid_scores, threshold)
        test_pred = multilabel_predictions_from_scores(test_scores, threshold)
        test_metrics = compute_multilabel_metrics(y_test, test_pred)
        test_metrics["threshold"] = float(threshold)
        export_model_outputs(
            model_dir,
            y_valid=y_valid,
            valid_scores=valid_scores,
            valid_pred=valid_pred,
            y_test=y_test,
            test_scores=test_scores,
            test_pred=test_pred,
            id2label=id2label,
        )
        bundle = {
            "artifact_type": "law_deep_multilabel",
            "task_mode": "law_article_multilabel",
            "model_type": model_name,
            "pretrained_model": args.pretrained_model,
            "checkpoint_path": str(best_path),
            "threshold": float(threshold),
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
                "selection_metric": args.selection_metric,
                "seed": args.seed,
            },
            "label2id": label2id,
            "id2label": {str(k): v for k, v in id2label.items()},
        }
        (model_dir / "model_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
        all_metrics[model_name] = {
            "best_valid": best_valid,
            "tuned_valid": tuned_valid,
            "test": test_metrics,
            "best_model_path": str(best_path),
            "model_bundle_path": str(model_dir / "model_bundle.json"),
            "threshold": float(threshold),
            "config": bundle["config"],
        }
        print(f"[Valid tuned] {tuned_valid}")
        print(f"[Test       ] {test_metrics}")

    (args.output_dir / "metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[Done] law deep models finished")


if __name__ == "__main__":
    main()
