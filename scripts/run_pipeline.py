#!/usr/bin/env python
"""一键执行：论文口径数据准备 + 深度平层 + 深度层次 + 结果表。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n[Run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paper-style charge prediction pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("../2018数据集/2018数据集"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed_110_paper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper"))
    parser.add_argument("--target-size", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--pretrained-model", type=str, default="hfl/chinese-bert-wwm-ext")
    parser.add_argument("--deep-models", nargs="+", default=["fc", "rcnn"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-flat", action="store_true")
    parser.add_argument("--skip-hier", action="store_true")
    parser.add_argument("--fallback-to-flat", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_prepare:
        run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "prepare_data.py"),
                "--data-dir",
                str(args.data_dir),
                "--output-dir",
                str(args.processed_dir),
                "--target-size",
                str(args.target_size),
                "--seed",
                str(args.seed),
            ]
        )

    if not args.skip_flat:
        run(
            [
                sys.executable,
                str(ROOT_DIR / "scripts" / "train_deep_models.py"),
                "--data-dir",
                str(args.processed_dir),
                "--output-dir",
                str(args.output_dir / "deep_models"),
                "--models",
                *args.deep_models,
                "--device",
                args.device,
                "--pretrained-model",
                args.pretrained_model,
                "--epochs",
                str(args.epochs),
                "--max-length",
                str(args.max_length),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--seed",
                str(args.seed),
            ]
        )

    if not args.skip_hier:
        for model_name in args.deep_models:
            fine_ckpt = args.output_dir / "deep_models" / model_name / f"best_{model_name}.pt"
            cmd = [
                sys.executable,
                str(ROOT_DIR / "scripts" / "train_deep_hierarchical.py"),
                "--data-dir",
                str(args.processed_dir),
                "--output-dir",
                str(args.output_dir / f"deep_hierarchical_{model_name}"),
                "--device",
                args.device,
                "--pretrained-model",
                args.pretrained_model,
                "--fine-model-type",
                model_name,
                "--coarse-model-type",
                model_name,
                "--epochs",
                str(args.epochs),
                "--max-length",
                str(args.max_length),
                "--train-batch-size",
                str(args.train_batch_size),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--gradient-accumulation-steps",
                str(args.gradient_accumulation_steps),
                "--seed",
                str(args.seed),
            ]
            if args.fallback_to_flat:
                cmd.append("--fallback-to-flat")
            if fine_ckpt.exists():
                cmd.extend(["--fine-checkpoint", str(fine_ckpt)])
            run(cmd)

    run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "make_results_table.py"),
            "--output-dir",
            str(args.output_dir),
            "--save-path",
            str(args.output_dir / "results_table.csv"),
        ]
    )

    print("\n[Done] Full paper-style pipeline finished")


if __name__ == "__main__":
    main()
