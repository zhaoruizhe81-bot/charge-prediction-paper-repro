#!/usr/bin/env python
"""构建离线可用的本地 BERT（随机初始化）与中文字符词表。"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from transformers import BertConfig, BertModel, BertTokenizerFast

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.data_utils import read_jsonl


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local random BERT assets")
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train_50k.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/local_bert"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    return parser.parse_args()


def build_vocab(train_file: Path, vocab_size: int) -> list[str]:
    records = read_jsonl(train_file)
    counter: collections.Counter[str] = collections.Counter()

    for item in records:
        text = item.get("fact", "")
        for ch in str(text):
            if ch.strip():
                counter[ch] += 1

    keep_size = max(0, vocab_size - len(SPECIAL_TOKENS))
    top_chars = [ch for ch, _ in counter.most_common(keep_size)]
    return SPECIAL_TOKENS + top_chars


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_vocab(args.train_file, args.vocab_size)
    vocab_path = args.output_dir / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")

    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(args.output_dir)

    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=512,
        pad_token_id=0,
    )
    model = BertModel(config)
    model.save_pretrained(args.output_dir)

    meta = {
        "vocab_size": len(vocab),
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "intermediate_size": args.intermediate_size,
        "note": "Randomly initialized local BERT for offline training.",
    }
    (args.output_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Local BERT assets built")
    print(f"output: {args.output_dir}")
    print(f"vocab_size: {len(vocab)}")


if __name__ == "__main__":
    main()
