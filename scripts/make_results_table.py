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


def dataframe_to_markdown(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return ""
    headers = [str(col) for col in dataframe.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in dataframe.iterrows():
        values: list[str] = []
        for col in dataframe.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def paper_f1(metrics: dict) -> float:
    if "f1_score" in metrics:
        return float(metrics.get("f1_score", 0.0))
    return float((float(metrics.get("f1_macro", 0.0)) + float(metrics.get("f1_micro", 0.0))) / 2.0)


def metric_row(task: str, model: str, variant: str, source: str, metrics: dict) -> dict[str, str | float]:
    return {
        "task": task,
        "family": "DL",
        "model": model,
        "variant": variant,
        "source": source,
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "recall_macro": float(metrics.get("recall_macro", 0.0)),
        "recall_micro": float(metrics.get("recall_micro", 0.0)),
        "f1_score": paper_f1(metrics),
    }


def collect_flat_rows(rows: list[dict[str, str | float]], output_dir: Path) -> None:
    metrics = load_json(output_dir / "deep_models" / "metrics.json")
    for model_name, model_metrics in metrics.items():
        rows.append(metric_row("charge_prediction", model_name, "flat", "deep_models", model_metrics.get("test", {})))


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
        rows.append(metric_row("charge_prediction", model_name, "flat", path.parent.name, metrics.get("test", {}).get("fine_flat", {})))
        rows.append(metric_row("charge_prediction", model_name, "hier", path.parent.name, metrics.get("test", {}).get("fine_hier", {})))
        for item in metrics.get("intermediate_rows", []):
            intermediate_rows.append(
                {
                    "task": "charge_prediction",
                    "model": model_name,
                    "source": path.parent.name,
                    "split": item.get("split", ""),
                    "stage": item.get("stage", ""),
                    "accuracy": float(item.get("accuracy", 0.0)),
                    "recall_macro": float(item.get("recall_macro", 0.0)),
                    "recall_micro": float(item.get("recall_micro", 0.0)),
                    "f1_score": float(item.get("f1_score", 0.0)),
                }
            )
    return intermediate_rows


def collect_law_rows(rows: list[dict[str, str | float]], output_dir: Path) -> list[dict[str, object]]:
    intermediate_rows: list[dict[str, object]] = []
    law_deep = load_json(output_dir / "law_deep" / "metrics.json")
    for model_name, model_metrics in law_deep.items():
        rows.append(metric_row("law_recommendation", model_name, "flat", "law_deep", model_metrics.get("test", {})))

    for path in sorted(output_dir.glob("law_hierarchical*/metrics.json")):
        law_hier = load_json(path)
        if not law_hier:
            continue
        source = path.parent.name
        suffix = "" if source == "law_hierarchical" else "_" + source.replace("law_hierarchical_", "")
        model_name = str(law_hier.get("config", {}).get("base_model", "law"))
        rows.append(metric_row("law_recommendation", model_name, "flat", source, law_hier.get("test", {}).get("flat_law", {})))
        for metric_key, variant_name in [
            ("fine_hier", "hier"),
            ("fine_hier_accuracy", "hier_accuracy"),
            ("fine_hier_f1", "hier_f1"),
        ]:
            if metric_key in law_hier.get("test", {}):
                rows.append(metric_row("law_recommendation", model_name, f"{variant_name}{suffix}", source, law_hier.get("test", {}).get(metric_key, {})))
        for item in law_hier.get("intermediate_rows", []):
            intermediate_rows.append(
                {
                    "task": "law_recommendation",
                    "model": model_name,
                    "source": source,
                    "split": item.get("split", ""),
                    "stage": item.get("stage", ""),
                    "accuracy": float(item.get("accuracy", 0.0)),
                    "recall_macro": float(item.get("recall_macro", 0.0)),
                    "recall_micro": float(item.get("recall_micro", 0.0)),
                    "f1_score": float(item.get("f1_score", 0.0)),
                }
            )
    return intermediate_rows


def deduplicate_best(rows: list[dict[str, str | float]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = (
        table.sort_values(by=["task", "model", "variant", "f1_score", "accuracy"], ascending=[True, True, True, False, False])
        .drop_duplicates(subset=["task", "model", "variant"], keep="first")
        .sort_values(by=["task", "accuracy", "f1_score"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return table


def build_contrast_table(table: pd.DataFrame) -> pd.DataFrame:
    contrast_rows: list[dict[str, str | float]] = []
    grouped = table.groupby(["task", "model"], dropna=False)
    for (task, model), group in grouped:
        flat_rows = group[group["variant"] == "flat"]
        hier_rows = group[group["variant"].astype(str).str.startswith("hier")]
        if flat_rows.empty or hier_rows.empty:
            continue
        flat = flat_rows.iloc[0]
        hier = hier_rows.iloc[0]
        contrast_rows.append(
            {
                "task": task,
                "family": "DL",
                "model": model,
                "base_variant": "flat",
                "hier_variant": str(hier["variant"]),
                "base_accuracy": float(flat["accuracy"]),
                "hier_accuracy": float(hier["accuracy"]),
                "delta_accuracy": float(hier["accuracy"] - flat["accuracy"]),
                "base_recall_macro": float(flat["recall_macro"]),
                "hier_recall_macro": float(hier["recall_macro"]),
                "delta_recall_macro": float(hier["recall_macro"] - flat["recall_macro"]),
                "base_recall_micro": float(flat["recall_micro"]),
                "hier_recall_micro": float(hier["recall_micro"]),
                "delta_recall_micro": float(hier["recall_micro"] - flat["recall_micro"]),
                "base_f1_score": float(flat["f1_score"]),
                "hier_f1_score": float(hier["f1_score"]),
                "delta_f1_score": float(hier["f1_score"] - flat["f1_score"]),
            }
        )
    if not contrast_rows:
        return pd.DataFrame()
    return pd.DataFrame(contrast_rows).sort_values(by=["task", "delta_accuracy", "delta_f1_score"], ascending=[True, False, False])


def build_best_contrast_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, group in table.groupby("task", dropna=False):
        flat_rows = group[group["variant"] == "flat"]
        hier_rows = group[group["variant"].astype(str).str.startswith("hier")]
        if flat_rows.empty or hier_rows.empty:
            continue
        flat = flat_rows.sort_values(by=["accuracy", "f1_score"], ascending=False).iloc[0]
        hier = hier_rows.sort_values(by=["accuracy", "f1_score"], ascending=False).iloc[0]
        rows.append(
            {
                "task": task,
                "base_model": flat["model"],
                "hier_model": hier["model"],
                "base_source": flat["source"],
                "hier_source": hier["source"],
                "base_accuracy": float(flat["accuracy"]),
                "hier_accuracy": float(hier["accuracy"]),
                "delta_accuracy": float(hier["accuracy"] - flat["accuracy"]),
                "base_recall_macro": float(flat["recall_macro"]),
                "hier_recall_macro": float(hier["recall_macro"]),
                "delta_recall_macro": float(hier["recall_macro"] - flat["recall_macro"]),
                "base_recall_micro": float(flat["recall_micro"]),
                "hier_recall_micro": float(hier["recall_micro"]),
                "delta_recall_micro": float(hier["recall_micro"] - flat["recall_micro"]),
                "base_f1_score": float(flat["f1_score"]),
                "hier_f1_score": float(hier["f1_score"]),
                "delta_f1_score": float(hier["f1_score"] - flat["f1_score"]),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["delta_accuracy", "delta_f1_score"], ascending=False)


def main() -> None:
    args = parse_args()

    rows: list[dict[str, str | float]] = []
    collect_flat_rows(rows, args.output_dir)
    intermediate_rows = collect_hier_rows(rows, args.output_dir)
    intermediate_rows.extend(collect_law_rows(rows, args.output_dir))

    if not rows:
        raise RuntimeError("No metrics found. Please run training first.")

    table = deduplicate_best(rows)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.save_path, index=False, encoding="utf-8-sig")
    (args.save_path.with_suffix(".md")).write_text(dataframe_to_markdown(table), encoding="utf-8")

    contrast = build_contrast_table(table)
    if not contrast.empty:
        contrast_path = args.save_path.with_name(f"{args.save_path.stem}_contrast.csv")
        contrast.to_csv(contrast_path, index=False, encoding="utf-8-sig")
        (contrast_path.with_suffix(".md")).write_text(dataframe_to_markdown(contrast), encoding="utf-8")

    best_contrast = build_best_contrast_table(table)
    if not best_contrast.empty:
        best_contrast_path = args.save_path.with_name(f"{args.save_path.stem}_best_contrast.csv")
        best_contrast.to_csv(best_contrast_path, index=False, encoding="utf-8-sig")
        (best_contrast_path.with_suffix(".md")).write_text(dataframe_to_markdown(best_contrast), encoding="utf-8")

    if intermediate_rows:
        intermediate = pd.DataFrame(intermediate_rows).sort_values(by=["task", "model", "split", "stage"])
        intermediate_path = args.save_path.with_name(f"{args.save_path.stem}_intermediate.csv")
        intermediate.to_csv(intermediate_path, index=False, encoding="utf-8-sig")
        (intermediate_path.with_suffix(".md")).write_text(dataframe_to_markdown(intermediate), encoding="utf-8")

    summary = {
        "best_accuracy": float(table["accuracy"].max()),
        "best_f1_score": float(table["f1_score"].max()),
        "num_models": int(table[["task", "model"]].drop_duplicates().shape[0]),
        "note": "Main table uses accuracy, macro recall, micro recall, and paper-style F1score.",
    }
    (args.save_path.with_name("summary.json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] Result tables generated")
    print(table)
    if not contrast.empty:
        print("\n[Contrast]")
        print(contrast)


if __name__ == "__main__":
    main()
