#!/usr/bin/env python
"""Minimal smoke checks for hierarchical FC training on Windows."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "processed_110_paper"
BERT_DIR = ROOT_DIR / "chinese-bert-wwm-ext"
FINE_CHECKPOINT = ROOT_DIR / "outputs_paper" / "deep_models" / "fc" / "best_fc.pt"


def ok(message: str) -> None:
    print(f"[OK] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing dataset split: {path}")
    return pd.read_json(path, lines=True)


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


def main() -> None:
    if not BERT_DIR.exists():
        fail(f"Missing local BERT directory: {BERT_DIR}")
    if not (BERT_DIR / "pytorch_model.bin").exists():
        fail(f"Missing BERT weight file: {BERT_DIR / 'pytorch_model.bin'}")
    ok(f"Found local BERT directory: {BERT_DIR}")

    if not FINE_CHECKPOINT.exists():
        fail(f"Missing flat FC checkpoint: {FINE_CHECKPOINT}")
    ok(f"Found flat FC checkpoint: {FINE_CHECKPOINT}")

    tokenizer = AutoTokenizer.from_pretrained(str(BERT_DIR))
    ok(f"Loaded tokenizer: {tokenizer.__class__.__name__}")

    if not torch.cuda.is_available():
        fail("CUDA is not available in the current environment.")
    ok(f"CUDA available: {torch.cuda.get_device_name(0)}")

    train_df = load_split(DATA_DIR / "train_50k.jsonl")
    valid_df = load_split(DATA_DIR / "valid_50k.jsonl")
    test_df = load_split(DATA_DIR / "test_50k.jsonl")

    fine_to_coarse = build_fine_to_coarse_mapping(train_df)
    train_df = normalize_coarse_labels(train_df, fine_to_coarse)
    valid_df = normalize_coarse_labels(valid_df, fine_to_coarse)
    test_df = normalize_coarse_labels(test_df, fine_to_coarse)

    coarse_labels = sorted(train_df["coarse_label"].unique().tolist())
    for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
        problems: list[str] = []
        for coarse_label in coarse_labels:
            train_labels = set(train_df.loc[train_df["coarse_label"] == coarse_label, "fine_label"].tolist())
            split_labels = set(split_df.loc[split_df["coarse_label"] == coarse_label, "fine_label"].tolist())
            extra_labels = sorted(split_labels - train_labels)
            if extra_labels:
                preview = ", ".join(extra_labels[:5])
                problems.append(f"{coarse_label}: {preview}")
        if problems:
            fail(f"{split_name} split still has labels outside local coarse buckets: {'; '.join(problems)}")
        ok(f"{split_name} split local coarse buckets are consistent")

    print("\n[Done] Hierarchical FC smoke checks passed. You can run run_train_hier_fc.cmd safely.")


if __name__ == "__main__":
    main()
