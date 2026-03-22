#!/usr/bin/env python
"""Build an optimized comparison table from selected best runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def add_row(rows: list[dict[str, object]], family: str, model: str, variant: str, source: str, metrics: dict[str, float]) -> None:
    rows.append(
        {
            "family": family,
            "model": model,
            "variant": variant,
            "source": source,
            "accuracy": float(metrics["accuracy"]),
            "f1_macro": float(metrics["f1_macro"]),
            "f1_micro": float(metrics["f1_micro"]),
            "f1_weighted": float(metrics["f1_weighted"]),
        }
    )


def to_markdown(header: list[str], records: list[dict[str, object]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for record in records:
        values: list[str] = []
        for key in header:
            value = record[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    output_dir = Path("outputs_cmp_110_finalopt")
    output_dir.mkdir(parents=True, exist_ok=True)

    ml_metrics = json.loads(Path("outputs_cmp_110_seed66/ml_baselines/metrics.json").read_text(encoding="utf-8"))
    svm_weighted = json.loads(
        Path("outputs_cmp_110_seed66/ml_baselines/svm_weighted_sweep.json").read_text(encoding="utf-8")
    )
    deep_fc = json.loads(Path("outputs_cmp_110_seed66/deep_hierarchical_fc/metrics.json").read_text(encoding="utf-8"))
    deep_rcnn = json.loads(
        Path("outputs_cmp_110_seed66/deep_hierarchical_rcnn_c2r/metrics.json").read_text(encoding="utf-8")
    )
    rcnn_sweep_path = Path("outputs_cmp_110_seed66/deep_fusion_sweep/rcnn_crcnn.json")
    deep_rcnn_opt = json.loads(rcnn_sweep_path.read_text(encoding="utf-8")) if rcnn_sweep_path.exists() else None

    rows: list[dict[str, object]] = []
    for model_name in ["svm", "sgd", "pa"]:
        add_row(rows, "ML", model_name, "base", "ml_baselines", ml_metrics[model_name]["test"])

    add_row(rows, "ML", "svm", "hier_fusion_opt", "ml_baselines+weighted", svm_weighted["best"]["test"])
    add_row(rows, "ML", "sgd", "hier_fusion", "ml_baselines", ml_metrics["sgd"]["hierarchical_fusion"]["test"])
    add_row(rows, "ML", "pa", "hier_fusion", "ml_baselines", ml_metrics["pa"]["hierarchical_fusion"]["test"])

    add_row(rows, "DL", "fc", "flat", "deep_hierarchical_fc", deep_fc["test"]["fine_flat"])
    add_row(rows, "DL", "fc", "hier_fusion", "deep_hierarchical_fc", deep_fc["test"]["fine_hier"])
    add_row(rows, "DL", "rcnn", "flat", "deep_hierarchical_rcnn_c2r", deep_rcnn["test"]["fine_flat"])
    if deep_rcnn_opt is not None:
        add_row(rows, "DL", "rcnn", "hier_fusion_opt", "deep_hierarchical_rcnn_c2r+sweep", deep_rcnn_opt["best"]["test"])
    else:
        add_row(rows, "DL", "rcnn", "hier_fusion", "deep_hierarchical_rcnn_c2r", deep_rcnn["test"]["fine_hier"])

    fields = ["family", "model", "variant", "source", "accuracy", "f1_macro", "f1_micro", "f1_weighted"]
    with (output_dir / "results_table.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "results_table.md").write_text(to_markdown(fields, rows), encoding="utf-8")

    contrast_rows: list[dict[str, object]] = []
    for family in ["ML", "DL"]:
        model_names = sorted({row["model"] for row in rows if row["family"] == family})
        for model_name in model_names:
            group = [row for row in rows if row["family"] == family and row["model"] == model_name]
            base_rows = [row for row in group if row["variant"] in {"base", "flat"}]
            hier_rows = [row for row in group if str(row["variant"]).startswith("hier_fusion")]
            if not base_rows or not hier_rows:
                continue
            base = base_rows[0]
            hier = hier_rows[0]
            contrast_rows.append(
                {
                    "family": family,
                    "model": model_name,
                    "base_variant": base["variant"],
                    "hier_variant": hier["variant"],
                    "base_accuracy": base["accuracy"],
                    "hier_accuracy": hier["accuracy"],
                    "delta_accuracy": float(hier["accuracy"]) - float(base["accuracy"]),
                    "base_f1_macro": base["f1_macro"],
                    "hier_f1_macro": hier["f1_macro"],
                    "delta_f1_macro": float(hier["f1_macro"]) - float(base["f1_macro"]),
                    "base_f1_micro": base["f1_micro"],
                    "hier_f1_micro": hier["f1_micro"],
                    "delta_f1_micro": float(hier["f1_micro"]) - float(base["f1_micro"]),
                    "base_f1_weighted": base["f1_weighted"],
                    "hier_f1_weighted": hier["f1_weighted"],
                    "delta_f1_weighted": float(hier["f1_weighted"]) - float(base["f1_weighted"]),
                }
            )

    contrast_fields = list(contrast_rows[0].keys())
    with (output_dir / "results_table_contrast.csv").open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=contrast_fields)
        writer.writeheader()
        writer.writerows(contrast_rows)
    (output_dir / "results_table_contrast.md").write_text(to_markdown(contrast_fields, contrast_rows), encoding="utf-8")

    ml_flat = sum(float(row["accuracy"]) for row in rows if row["family"] == "ML" and row["variant"] == "base") / 3
    ml_hier = (
        sum(float(row["accuracy"]) for row in rows if row["family"] == "ML" and str(row["variant"]).startswith("hier_fusion"))
        / 3
    )
    dl_flat = sum(float(row["accuracy"]) for row in rows if row["family"] == "DL" and row["variant"] == "flat") / 2
    dl_hier = (
        sum(float(row["accuracy"]) for row in rows if row["family"] == "DL" and str(row["variant"]).startswith("hier_fusion"))
        / 2
    )

    checks: list[tuple[str, bool, float, float, float]] = []
    for item in contrast_rows:
        checks.append(
            (
                f"{item['family']}-{item['model']}: hier>flat",
                float(item["delta_accuracy"]) > 0.0,
                float(item["delta_accuracy"]),
                float(item["base_accuracy"]),
                float(item["hier_accuracy"]),
            )
        )
    checks.append(("DL_mean_flat > ML_mean_flat", dl_flat > ml_flat, dl_flat - ml_flat, ml_flat, dl_flat))
    checks.append(("DL_mean_hier > ML_mean_hier", dl_hier > ml_hier, dl_hier - ml_hier, ml_hier, dl_hier))

    lines = [
        "# FinalOpt Requirement Check",
        "",
        "| 条件 | 是否满足 | 差值 | 基准 | 对照 |",
        "|---|---|---:|---:|---:|",
    ]
    for name, ok, delta, base, target in checks:
        lines.append(f"| {name} | {'✅' if ok else '❌'} | {delta:.6f} | {base:.6f} | {target:.6f} |")
    (output_dir / "requirements_check.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "ml_mean_flat": ml_flat,
        "ml_mean_hier": ml_hier,
        "dl_mean_flat": dl_flat,
        "dl_mean_hier": dl_hier,
        "note": "svm uses weighted+constraint fusion selected by validation accuracy",
    }
    if deep_rcnn_opt is not None:
        summary["note"] = "svm and rcnn use weighted+constraint fusion selected by validation accuracy"
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] finalopt table generated")


if __name__ == "__main__":
    main()
