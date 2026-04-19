#!/usr/bin/env python
"""Build final tables and ROC/AUC figures from completed experiment outputs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show final parameters, metrics, and ROC/AUC curves")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_paper"))
    parser.add_argument("--export-dir", type=Path, default=Path("outputs_paper/final_report"))
    parser.add_argument("--skip-table-refresh", action="store_true")
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def safe_name(path: Path) -> str:
    parts = [part for part in path.with_suffix("").parts[-3:] if part not in {"/", "\\"}]
    raw = "_".join(parts).replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw).strip("_") or "roc_auc"


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


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, save_path: Path, title: str) -> dict[str, object]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    if y_true.ndim == 1:
        classes = np.arange(scores.shape[1])
        y_binary = label_binarize(y_true, classes=classes)
    else:
        y_binary = y_true.astype(int)

    fpr, tpr, _ = roc_curve(y_binary.ravel(), scores.ravel())
    micro_auc = float(auc(fpr, tpr))
    try:
        macro_auc = float(roc_auc_score(y_binary, scores, average="macro"))
    except ValueError:
        macro_auc = 0.0

    plt.figure(figsize=(7.2, 5.2))
    plt.plot(fpr, tpr, label=f"micro AUC = {micro_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()
    return {"micro_auc": micro_auc, "macro_auc": macro_auc, "figure": str(save_path)}


def collect_eval_outputs(output_dir: Path, export_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for npz_path in sorted(output_dir.glob("**/eval_outputs.npz")):
        with np.load(npz_path) as data:
            y_test = data["y_test"]
            if "test_scores" in data:
                scores = data["test_scores"]
            elif "test_logits" in data:
                scores = softmax(data["test_logits"])
            else:
                continue
        title = npz_path.parent.relative_to(output_dir).as_posix()
        figure_path = export_dir / "roc_auc" / f"{safe_name(npz_path.parent)}.png"
        try:
            auc_info = plot_roc_curve(y_test, scores, figure_path, title)
        except Exception as exc:
            auc_info = {"micro_auc": 0.0, "macro_auc": 0.0, "figure": "", "error": repr(exc)}
        rows.append(
            {
                "source": title,
                "samples": int(y_test.shape[0]),
                "labels": int(scores.shape[1]),
                **auc_info,
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table.to_csv(export_dir / "auc_summary.csv", index=False, encoding="utf-8-sig")
    return table


def refresh_result_tables(output_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT_DIR / "scripts" / "make_results_table.py"),
            "--output-dir",
            str(output_dir),
            "--save-path",
            str(output_dir / "results_table.csv"),
        ],
        check=False,
    )


def build_markdown(output_dir: Path, export_dir: Path, auc_table: pd.DataFrame) -> None:
    lines = ["# Final Experiment Report", ""]
    result_path = output_dir / "results_table.csv"
    contrast_path = output_dir / "results_table_contrast.csv"
    intermediate_path = output_dir / "results_table_intermediate.csv"
    summary_path = output_dir / "summary.json"

    if result_path.exists():
        result_table = pd.read_csv(result_path)
        lines.extend(["## Main Results", "", dataframe_to_markdown(result_table), ""])
    if contrast_path.exists():
        contrast_table = pd.read_csv(contrast_path)
        lines.extend(["## Flat vs Hierarchical Contrast", "", dataframe_to_markdown(contrast_table), ""])
    if intermediate_path.exists():
        intermediate_table = pd.read_csv(intermediate_path)
        lines.extend(["## Process Metrics", "", dataframe_to_markdown(intermediate_table), ""])
    if not auc_table.empty:
        lines.extend(["## ROC/AUC", "", dataframe_to_markdown(auc_table), ""])
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        lines.extend(["## Summary JSON", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    lines.extend(
        [
            "## Output Files",
            "",
            f"- Main table: `{result_path}`",
            f"- Contrast table: `{contrast_path}`",
            f"- Intermediate table: `{intermediate_path}`",
            f"- AUC summary: `{export_dir / 'auc_summary.csv'}`",
            f"- ROC figures: `{export_dir / 'roc_auc'}`",
        ]
    )
    (export_dir / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_requirement_check(output_dir: Path, export_dir: Path, auc_table: pd.DataFrame) -> None:
    checks: list[dict[str, object]] = []
    result_path = output_dir / "results_table.csv"
    contrast_path = output_dir / "results_table_contrast.csv"
    intermediate_path = output_dir / "results_table_intermediate.csv"

    if result_path.exists():
        result_table = pd.read_csv(result_path)
        expected_cols = {"accuracy", "recall_macro", "recall_micro", "f1_score"}
        checks.append(
            {
                "requirement": "Main table uses accuracy, two recall columns, and F1score",
                "status": "PASS" if expected_cols.issubset(result_table.columns) else "FAIL",
                "detail": ",".join([col for col in ["accuracy", "recall_macro", "recall_micro", "f1_score"] if col in result_table.columns]),
            }
        )
    else:
        checks.append({"requirement": "Main result table exists", "status": "FAIL", "detail": str(result_path)})

    if contrast_path.exists():
        contrast = pd.read_csv(contrast_path)
        if "delta_accuracy" in contrast.columns:
            max_delta = float(contrast["delta_accuracy"].max()) if not contrast.empty else 0.0
            checks.append(
                {
                    "requirement": "Best hierarchical accuracy gain is at least 2%",
                    "status": "PASS" if max_delta >= 0.02 else "CHECK",
                    "detail": f"max_delta_accuracy={max_delta:.6f}",
                }
            )
        if "delta_f1_score" in contrast.columns:
            max_delta = float(contrast["delta_f1_score"].max()) if not contrast.empty else 0.0
            checks.append(
                {
                    "requirement": "Best hierarchical F1score gain is at least 2%",
                    "status": "PASS" if max_delta >= 0.02 else "CHECK",
                    "detail": f"max_delta_f1_score={max_delta:.6f}",
                }
            )
    else:
        checks.append({"requirement": "Flat vs hierarchical contrast table exists", "status": "FAIL", "detail": str(contrast_path)})

    checks.append(
        {
            "requirement": "Intermediate process metrics exist",
            "status": "PASS" if intermediate_path.exists() else "FAIL",
            "detail": str(intermediate_path),
        }
    )
    checks.append(
        {
            "requirement": "ROC/AUC outputs exist",
            "status": "PASS" if not auc_table.empty else "FAIL",
            "detail": f"num_curves={len(auc_table)}",
        }
    )

    check_table = pd.DataFrame(checks)
    check_table.to_csv(export_dir / "requirements_check.csv", index=False, encoding="utf-8-sig")
    lines = ["# Requirement Check", "", dataframe_to_markdown(check_table), ""]
    (export_dir / "requirements_check.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.export_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_table_refresh:
        refresh_result_tables(args.output_dir)
    auc_table = collect_eval_outputs(args.output_dir, args.export_dir)
    build_markdown(args.output_dir, args.export_dir, auc_table)
    write_requirement_check(args.output_dir, args.export_dir, auc_table)
    print("[Done] final report generated")
    print(f"  summary: {args.export_dir / 'final_summary.md'}")
    if not auc_table.empty:
        print(f"  auc: {args.export_dir / 'auc_summary.csv'}")


if __name__ == "__main__":
    main()
