#!/usr/bin/env python
"""Smoke checks for flat FC training on Windows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed_110_paper"
BERT_DIR = ROOT_DIR / "chinese-bert-wwm-ext"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.deep_models import (  # noqa: E402
    DeepChargeTrainer,
    DeepTrainingConfig,
    build_dataloaders,
    build_tokenizer_cache_key,
    resolve_device,
)


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing dataset split: {path}")
    return pd.read_json(path, lines=True)


def main() -> None:
    if not BERT_DIR.exists():
        fail(f"Missing local BERT directory: {BERT_DIR}")
    if not (BERT_DIR / "pytorch_model.bin").exists():
        fail(f"Missing BERT weight file: {BERT_DIR / 'pytorch_model.bin'}")
    ok(f"Found local BERT directory: {BERT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))
    ok(f"Loaded tokenizer: {tokenizer.__class__.__name__}")

    device = resolve_device("cuda")
    if device.type != "cuda":
        fail("CUDA is not available in the current environment.")
    ok(f"CUDA available: {torch.cuda.get_device_name(0)}")

    train_path = DATA_DIR / "train_50k.jsonl"
    valid_path = DATA_DIR / "valid_50k.jsonl"
    test_path = DATA_DIR / "test_50k.jsonl"
    train_df = load_split(train_path).head(32).reset_index(drop=True)
    valid_df = load_split(valid_path).head(16).reset_index(drop=True)
    test_df = load_split(test_path).head(16).reset_index(drop=True)

    labels = sorted(train_df["fine_label"].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(labels)}
    y_train = train_df["fine_label"].map(label2id).astype(int).to_numpy()
    valid_mask = valid_df["fine_label"].isin(label2id).to_numpy()
    test_mask = test_df["fine_label"].isin(label2id).to_numpy()
    valid_df = valid_df.loc[valid_mask].reset_index(drop=True)
    test_df = test_df.loc[test_mask].reset_index(drop=True)
    y_valid = valid_df["fine_label"].map(label2id).astype(int).to_numpy()
    y_test = test_df["fine_label"].map(label2id).astype(int).to_numpy()

    config = DeepTrainingConfig(
        pretrained_model_name=str(BERT_DIR),
        max_length=256,
        train_batch_size=2,
        eval_batch_size=2,
        num_workers=0,
        pin_memory=True,
        non_blocking=True,
        enable_tokenizer_cache=True,
        tokenizer_cache_dir=str(ROOT_DIR / ".cache" / "tokenized"),
    )

    train_cache_key = build_tokenizer_cache_key(train_path, config, extra="flat-smoke-train-32")
    valid_cache_key = build_tokenizer_cache_key(valid_path, config, extra="flat-smoke-valid-16")
    test_cache_key = build_tokenizer_cache_key(test_path, config, extra="flat-smoke-test-16")

    train_loader, valid_loader, test_loader, _ = build_dataloaders(
        train_texts=train_df["fact"].tolist(),
        train_labels=y_train,
        valid_texts=valid_df["fact"].tolist(),
        valid_labels=y_valid,
        test_texts=test_df["fact"].tolist(),
        test_labels=y_test,
        config=config,
        train_cache_key=train_cache_key,
        valid_cache_key=valid_cache_key,
        test_cache_key=test_cache_key,
    )
    ok("Built dataloaders and tokenizer cache entries")

    trainer = DeepChargeTrainer("fc", max(int(np.max(y_train)) + 1, 1), config, device)
    batch = next(iter(train_loader))
    labels_tensor = batch.pop("labels").to(device, non_blocking=True)
    inputs = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    with torch.no_grad():
        logits = trainer.model(**inputs)
    if logits.shape[0] != labels_tensor.shape[0]:
        fail("Model forward batch size does not match labels.")
    ok(f"Ran one FC forward pass on CUDA: logits shape = {tuple(logits.shape)}")

    _ = trainer.evaluate(valid_loader)
    _ = trainer.collect_logits(test_loader)
    ok("Validation and test smoke pass completed")

    print("\n[Done] Flat FC smoke checks passed. You can run run_train_flat_fc_optimized.cmd safely.")


if __name__ == "__main__":
    main()
