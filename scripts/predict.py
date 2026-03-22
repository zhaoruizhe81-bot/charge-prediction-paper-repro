#!/usr/bin/env python
"""加载训练好的模型进行罪名预测，支持深度平层/层次模型和旧版 joblib。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from charge_prediction.deep_models import (
    DeepChargeTrainer,
    DeepTrainingConfig,
    build_predict_dataloader,
    resolve_device,
)
from charge_prediction.fusion import hierarchical_constrained_decode, score_matrix_from_estimator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict accusation from fact text")
    parser.add_argument("--artifact-path", type=Path, required=True, help="Path to model bundle or output dir")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--text", type=str, default="", help="Single fact text")
    parser.add_argument("--input-file", type=Path, default=None, help="TXT/JSONL input file")
    parser.add_argument("--output-file", type=Path, default=None, help="Optional CSV/JSON output path")
    return parser.parse_args()


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True).clip(min=1e-12)


def load_inputs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.text:
        return [{"text": args.text}]
    if args.input_file is None:
        raise ValueError("Either --text or --input-file is required.")

    suffix = args.input_file.suffix.lower()
    rows: list[dict[str, str]] = []
    if suffix == ".jsonl":
        for line in args.input_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({"text": str(obj.get("fact") or obj.get("text") or "")})
    else:
        for line in args.input_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append({"text": line.strip()})
    return [row for row in rows if row["text"]]


def write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_bundle(path: Path) -> dict:
    if path.is_dir():
        bundle_path = path / "model_bundle.json"
    else:
        bundle_path = path
    if bundle_path.suffix == ".joblib":
        return {"legacy_joblib": True, "path": str(bundle_path)}
    return json.loads(bundle_path.read_text(encoding="utf-8"))


def build_config_from_bundle(bundle: dict) -> DeepTrainingConfig:
    return DeepTrainingConfig(
        pretrained_model_name=bundle["pretrained_model"],
        max_length=int(bundle.get("max_length", bundle.get("config", {}).get("max_length", 256))),
        eval_batch_size=int(bundle.get("eval_batch_size", bundle.get("config", {}).get("eval_batch_size", 16))),
        dropout=float(bundle.get("dropout", bundle.get("config", {}).get("dropout", 0.2))),
        rcnn_hidden_size=int(bundle.get("rcnn_hidden_size", bundle.get("config", {}).get("rcnn_hidden_size", 256))),
        rcnn_num_layers=int(bundle.get("rcnn_num_layers", bundle.get("config", {}).get("rcnn_num_layers", 1))),
        num_workers=int(bundle.get("num_workers", bundle.get("config", {}).get("num_workers", 0))),
    )


def load_deep_trainer(
    model_type: str,
    num_labels: int,
    config: DeepTrainingConfig,
    device: torch.device,
    checkpoint_path: str,
) -> DeepChargeTrainer:
    trainer = DeepChargeTrainer(model_type=model_type, num_labels=num_labels, config=config, device=device)
    trainer.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return trainer


def predict_deep_flat(bundle: dict, texts: list[str], device: torch.device) -> list[dict[str, object]]:
    config = build_config_from_bundle(bundle)
    id2label = {int(k): v for k, v in bundle["id2label"].items()}
    trainer = load_deep_trainer(
        model_type=bundle["model_type"],
        num_labels=len(id2label),
        config=config,
        device=device,
        checkpoint_path=bundle["checkpoint_path"],
    )
    logits = trainer.collect_logits(build_predict_dataloader(texts, config=config))
    probs = softmax(logits)
    preds = np.argmax(probs, axis=1)
    rows = []
    for idx, pred in enumerate(preds.tolist()):
        rows.append(
            {
                "text": texts[idx],
                "predicted_charge": id2label[int(pred)],
                "confidence": float(np.max(probs[idx])),
            }
        )
    return rows


def predict_deep_hier(bundle: dict, texts: list[str], device: torch.device) -> list[dict[str, object]]:
    config = build_config_from_bundle(bundle)
    fine_id2label = {int(k): v for k, v in bundle["fine_id2label"].items()}
    coarse_id2label = {int(k): v for k, v in bundle["coarse_id2label"].items()}

    flat_trainer = load_deep_trainer(
        model_type=bundle["fine_model_type"],
        num_labels=len(fine_id2label),
        config=config,
        device=device,
        checkpoint_path=bundle["flat_checkpoint"],
    )
    coarse_trainer = load_deep_trainer(
        model_type=bundle["coarse_model_type"],
        num_labels=len(coarse_id2label),
        config=config,
        device=device,
        checkpoint_path=bundle["coarse_checkpoint"],
    )

    flat_logits = flat_trainer.collect_logits(build_predict_dataloader(texts, config=config))
    coarse_logits = coarse_trainer.collect_logits(build_predict_dataloader(texts, config=config))

    flat_probs = softmax(flat_logits)
    coarse_probs = softmax(coarse_logits)
    flat_pred = np.argmax(flat_probs, axis=1)
    coarse_pred = np.argmax(coarse_probs, axis=1)

    routed_pred = np.array(flat_pred, copy=True)
    for coarse_id_str, model_info in bundle["local_models"].items():
        coarse_id = int(coarse_id_str)
        indices = np.where(coarse_pred == coarse_id)[0]
        if len(indices) == 0:
            continue
        subset_texts = [texts[idx] for idx in indices.tolist()]
        local_id2global = {int(k): int(v) for k, v in model_info["local_id2global_fine_id"].items()}
        local_trainer = load_deep_trainer(
            model_type=bundle["fine_model_type"],
            num_labels=len(local_id2global),
            config=config,
            device=device,
            checkpoint_path=model_info["checkpoint"],
        )
        local_logits = local_trainer.collect_logits(build_predict_dataloader(subset_texts, config=config))
        local_pred = np.argmax(local_logits, axis=1)
        routed_pred[indices] = np.asarray([local_id2global[int(item)] for item in local_pred], dtype=int)

    routing_config = bundle.get("routing_config", {})
    coarse_confidence = np.max(coarse_probs, axis=1)
    if coarse_probs.shape[1] <= 1:
        coarse_margin = np.full(coarse_confidence.shape[0], np.inf, dtype=float)
    else:
        sorted_probs = np.sort(coarse_probs, axis=1)
        coarse_margin = sorted_probs[:, -1] - sorted_probs[:, -2]

    use_hier = bool(routing_config.get("use_hierarchical_routing", False))
    fallback_to_flat = bool(routing_config.get("fallback_to_flat", False))
    conf_th = float(routing_config.get("coarse_threshold") or 0.0)
    margin_th = float(routing_config.get("coarse_margin_threshold") or 0.0)

    if use_hier and fallback_to_flat:
        use_route_mask = (coarse_confidence >= conf_th) & (coarse_margin >= margin_th)
        final_pred = np.where(use_route_mask, routed_pred, flat_pred)
    elif use_hier:
        use_route_mask = np.ones(len(texts), dtype=bool)
        final_pred = routed_pred
    else:
        use_route_mask = np.zeros(len(texts), dtype=bool)
        final_pred = flat_pred

    rows = []
    for idx, pred in enumerate(final_pred.tolist()):
        rows.append(
            {
                "text": texts[idx],
                "predicted_coarse": coarse_id2label[int(coarse_pred[idx])],
                "predicted_charge": fine_id2label[int(pred)],
                "flat_charge": fine_id2label[int(flat_pred[idx])],
                "routed_charge": fine_id2label[int(routed_pred[idx])],
                "coarse_confidence": float(coarse_confidence[idx]),
                "coarse_margin": float(coarse_margin[idx]),
                "used_hierarchical": bool(use_route_mask[idx]),
            }
        )
    return rows


def predict_legacy_joblib(path: Path, texts: list[str]) -> list[dict[str, object]]:
    bundle = joblib.load(path)
    rows: list[dict[str, object]] = []

    if "pipeline" in bundle:
        pipeline = bundle["pipeline"]
        id2label = bundle["id2label"]
        preds = pipeline.predict(texts)
        for idx, pred in enumerate(preds.tolist()):
            rows.append({"text": texts[idx], "predicted_charge": id2label[int(pred)]})
        return rows

    if "fine_pipeline" in bundle and "coarse_pipeline" in bundle:
        fine_pipeline = bundle["fine_pipeline"]
        coarse_pipeline = bundle["coarse_pipeline"]
        fine_id2label = bundle["fine_id2label"]
        coarse_id2label = bundle["coarse_id2label"]
        fine_label2id = bundle["fine_label2id"]
        coarse_label2id = bundle["coarse_label2id"]
        coarse_to_fine_raw = bundle.get("coarse_to_fine", {})
        coarse_to_fine = {int(k): set(int(v) for v in values) for k, values in coarse_to_fine_raw.items()}
        fusion_config = bundle.get("fusion_config", {})

        fine_scores = score_matrix_from_estimator(fine_pipeline, texts, num_classes=len(fine_label2id))
        coarse_scores = score_matrix_from_estimator(coarse_pipeline, texts, num_classes=len(coarse_label2id))
        coarse_pred = np.argmax(coarse_scores, axis=1)

        if bool(fusion_config.get("use_hier_fusion", False)):
            fine_pred = hierarchical_constrained_decode(
                fine_scores=fine_scores,
                coarse_scores=coarse_scores,
                coarse_to_fine=coarse_to_fine,
                top_k_coarse=int(fusion_config.get("top_k_coarse", 1)),
                confidence_threshold=float(fusion_config.get("confidence_threshold", 0.0)),
                max_fine_margin=float(fusion_config.get("max_fine_margin", float("inf"))),
            )
        else:
            fine_pred = np.argmax(fine_scores, axis=1)

        for idx, pred in enumerate(fine_pred.tolist()):
            rows.append(
                {
                    "text": texts[idx],
                    "predicted_coarse": coarse_id2label[int(coarse_pred[idx])],
                    "predicted_charge": fine_id2label[int(pred)],
                }
            )
        return rows

    if "classifier" in bundle:
        classifier = bundle["classifier"]
        fine_id2label = bundle["fine_id2label"]
        coarse_id2label = bundle["coarse_id2label"]
        coarse_pred, fine_pred = classifier.predict(texts)
        for idx, pred in enumerate(fine_pred.tolist()):
            rows.append(
                {
                    "text": texts[idx],
                    "predicted_coarse": coarse_id2label[int(coarse_pred[idx])],
                    "predicted_charge": fine_id2label[int(pred)],
                }
            )
        return rows

    raise RuntimeError("Unsupported model bundle format")


def main() -> None:
    args = parse_args()
    rows_in = load_inputs(args)
    texts = [row["text"] for row in rows_in]
    bundle = load_bundle(args.artifact_path)

    if bundle.get("legacy_joblib"):
        rows = predict_legacy_joblib(Path(bundle["path"]), texts)
    else:
        device = resolve_device(args.device)
        artifact_type = bundle.get("artifact_type", "")
        if artifact_type == "deep_flat":
            rows = predict_deep_flat(bundle, texts, device)
        elif artifact_type == "deep_hierarchical":
            rows = predict_deep_hier(bundle, texts, device)
        else:
            raise RuntimeError(f"Unsupported artifact type: {artifact_type}")

    if args.output_file is not None:
        write_rows(rows, args.output_file)

    for row in rows[:10]:
        print(json.dumps(row, ensure_ascii=False))
    if len(rows) > 10:
        print(f"... total={len(rows)}")


if __name__ == "__main__":
    main()
