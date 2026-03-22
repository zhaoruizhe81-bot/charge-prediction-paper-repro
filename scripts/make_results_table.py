#!/usr/bin/env python
"""汇总平层与层次深度模型结果，生成论文可用表格。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build result table from metrics")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper"))
    parser.add_argument("--save-path", type=Path, default=Path("outputs_paper/results_table.csv"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_row(model: str, variant: str, source: str, metrics: dict) -> dict[str, str | float]:
    return {
        "family": "DL",
        "model": model,
        "variant": variant,
        "source": source,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "f1_micro": float(metrics.get("f1_micro", 0.0)),
        "f1_weighted": float(metrics.get("f1_weighted", 0.0)),
    }


def collect_flat_rows(rows: list[dict[str, str | float]], output_dir: Path) -> None:
    metrics = load_json(output_dir / "deep_models" / "metrics.json")
    for model_name, model_metrics in metrics.items():
        rows.append(metric_row(model_name, "flat", "deep_models", model_metrics.get("test", {})))


def infer_model_name(path: Path, metrics: dict) -> str:
    configured = metrics.get("config", {}).get("fine_model_type")
    if configured:
        return str(configured)
    name = path.parent.name
    if name.startswith("deep_hierarchical_"):
        return name.replace("deep_hierarchical_", "", 1)
    return name


def collect_hier_rows(rows: list[dict[str, str | float]], output_dir: Path) -> list[dict[str, object]]:
    intermediate_rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("deep_hierarchical*/metrics.json")):
        metrics = load_json(path)
        model_name = infer_model_name(path, metrics)
        rows.append(metric_row(model_name, "flat", path.parent.name, metrics.get("test", {}).get("fine_flat", {})))
        rows.append(metric_row(model_name, "hier", path.parent.name, metrics.get("test", {}).get("fine_hier", {})))
        for item in metrics.get("intermediate_rows", []):
            intermediate_rows.append(
                {
                    "model": model_name,
                    "source": path.parent.name,
                    "split": item.get("split", ""),
                    "stage": item.get("stage", ""),
                    "accuracy": float(item.get("accuracy", 0.0)),
                }
            )
    return intermediate_rows


def deduplicate_best(rows: list[dict[str, str | float]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = (
        table.sort_values(by=["model", "variant", "f1_macro", "accuracy"], ascending=[True, True, False, False])
        .drop_duplicates(subset=["model", "variant"], keep="first")
        .sort_values(by=["accuracy", "f1_macro"], ascending=False)
        .reset_index(drop=True)
    )
    return table


def build_contrast_table(table: pd.DataFrame) -> pd.DataFrame:
    contrast_rows: list[dict[str, str | float]] = []
    grouped = table.groupby("model", dropna=False)
    for model, group in grouped:
        flat_rows = group[group["variant"] == "flat"]
        hier_rows = group[group["variant"] == "hier"]
        if flat_rows.empty or hier_rows.empty:
            continue
        flat = flat_rows.iloc[0]
        hier = hier_rows.iloc[0]
        contrast_rows.append(
            {
                "family": "DL",
                "model": model,
                "base_variant": "flat",
                "hier_variant": "hier",
                "base_accuracy": float(flat["accuracy"]),
                "hier_accuracy": float(hier["accuracy"]),
                "delta_accuracy": float(hier["accuracy"] - flat["accuracy"]),
                "base_f1_macro": float(flat["f1_macro"]),
                "hier_f1_macro": float(hier["f1_macro"]),
                "delta_f1_macro": float(hier["f1_macro"] - flat["f1_macro"]),
                "base_f1_micro": float(flat["f1_micro"]),
                "hier_f1_micro": float(hier["f1_micro"]),
                "delta_f1_micro": float(hier["f1_micro"] - flat["f1_micro"]),
                "base_f1_weighted": float(flat["f1_weighted"]),
                "hier_f1_weighted": float(hier["f1_weighted"]),
                "delta_f1_weighted": float(hier["f1_weighted"] - flat["f1_weighted"]),
            }
        )
    if not contrast_rows:
        return pd.DataFrame()
    return pd.DataFrame(contrast_rows).sort_values(by=["delta_accuracy", "delta_f1_macro"], ascending=False)


def main() -> None:
    args = parse_args()

    rows: list[dict[str, str | float]] = []
    collect_flat_rows(rows, args.output_dir)
    intermediate_rows = collect_hier_rows(rows, args.output_dir)

    if not rows:
        raise RuntimeError("No metrics found. Please run training first.")

    table = deduplicate_best(rows)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.save_path, index=False, encoding="utf-8-sig")
    (args.save_path.with_suffix(".md")).write_text(table.to_markdown(index=False), encoding="utf-8")

    contrast = build_contrast_table(table)
    if not contrast.empty:
        contrast_path = args.save_path.with_name(f"{args.save_path.stem}_contrast.csv")
        contrast.to_csv(contrast_path, index=False, encoding="utf-8-sig")
        (contrast_path.with_suffix(".md")).write_text(contrast.to_markdown(index=False), encoding="utf-8")

    if intermediate_rows:
        intermediate = pd.DataFrame(intermediate_rows).sort_values(by=["model", "split", "stage"])
        intermediate_path = args.save_path.with_name(f"{args.save_path.stem}_intermediate.csv")
        intermediate.to_csv(intermediate_path, index=False, encoding="utf-8-sig")
        (intermediate_path.with_suffix(".md")).write_text(intermediate.to_markdown(index=False), encoding="utf-8")

    summary = {
        "best_accuracy": float(table["accuracy"].max()),
        "num_models": int(table["model"].nunique()),
        "note": "Main table keeps the best flat and best hierarchical result per backbone.",
    }
    (args.save_path.with_name("summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Result tables generated")
    print(table)
    if not contrast.empty:
        print("\n[Contrast]")
        print(contrast)


if __name__ == "__main__":
    main()
