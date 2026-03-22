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
from charge_prediction.deep_models import DeepChargeTrainer, DeepTrainingConfig, build_dataloaders, resolve_device, set_seed


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_df = load_split(args.data_dir / "train_50k.jsonl")
    valid_df = load_split(args.data_dir / "valid_50k.jsonl")
    test_df = load_split(args.data_dir / "test_50k.jsonl")

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

    config = DeepTrainingConfig(
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

    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        train_texts=x_train,
        train_labels=y_train,
        valid_texts=x_valid,
        valid_labels=y_valid,
        test_texts=x_test,
        test_labels=y_test,
        config=config,
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
        )

        print(f"\n[Training] {model_name} on {device}")
        best_valid_metrics, best_model_path = trainer.fit(
            train_loader=train_loader,
            valid_loader=valid_loader,
            output_dir=model_output_dir,
        )
        test_metrics = trainer.evaluate(test_loader)

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
