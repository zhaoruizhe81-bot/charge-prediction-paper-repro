"""深度学习模型：BERT+FC / BERT+RCNN。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

from .metrics import compute_classification_metrics


@dataclass
class DeepTrainingConfig:
    pretrained_model_name: str = "hfl/chinese-bert-wwm-ext"
    max_length: int = 256
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    epochs: int = 3
    early_stopping_patience: int = 2
    dropout: float = 0.2
    rcnn_hidden_size: int = 256
    rcnn_num_layers: int = 1
    gradient_accumulation_steps: int = 1
    seed: int = 42
    num_workers: int = 2
    selection_metric: str = "accuracy"
    optimize_profile: str = "baseline"
    loss_name: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    sampler_name: str = "none"
    pin_memory: bool | None = None
    persistent_workers: bool | None = None
    prefetch_factor: int | None = None
    non_blocking: bool | None = None
    enable_amp: bool | None = None
    enable_tokenizer_cache: bool = False
    tokenizer_cache_dir: str = ".cache/tokenized"


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text": self.texts[index],
            "label": int(self.labels[index]),
        }


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: np.ndarray) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["label"] = int(self.labels[index])
        return item


class BatchTokenizerCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        texts = [item["text"] for item in batch]
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        tokenized["labels"] = labels
        return tokenized


class EncodedBatchCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        features = [{key: value for key, value in item.items() if key != "label"} for item in batch]
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        padded["labels"] = labels
        return padded


class FocalCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.weight,
            label_smoothing=self.label_smoothing,
        )
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8, max=1.0)
        focal_factor = torch.pow(1.0 - pt, self.gamma)
        return torch.mean(focal_factor * ce_loss)


def resolve_device(device_name: str = "auto") -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device: {device_name}")


def load_tokenizer(pretrained_model_name: str) -> Any:
    try:
        return AutoTokenizer.from_pretrained(pretrained_model_name)
    except OSError:
        pretrained_dir = Path(pretrained_model_name)
        vocab_path = pretrained_dir / "vocab.txt"
        if not vocab_path.exists():
            raise
        return BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=False)


def load_encoder_model(pretrained_model_name: str) -> nn.Module:
    model_dir = Path(pretrained_model_name)
    if model_dir.exists():
        weight_path = model_dir / "pytorch_model.bin"
        if weight_path.exists():
            try:
                state_dict = torch.load(weight_path, map_location="cpu")
                first_key = next(iter(state_dict.keys()), "")
                if first_key.startswith("bert."):
                    return BertModel.from_pretrained(pretrained_model_name)
            except Exception:
                pass
    return AutoModel.from_pretrained(pretrained_model_name)


class BertFCClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.bert = load_encoder_model(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls_feature = outputs.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls_feature))


class BertRCNNClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        num_labels: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.bert = load_encoder_model(pretrained_model_name)
        bert_hidden = self.bert.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(bert_hidden + hidden_size * 2, bert_hidden)
        self.fusion = nn.Linear(bert_hidden * 3, bert_hidden)
        self.norm = nn.LayerNorm(bert_hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_hidden, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        bert_features = outputs.last_hidden_state
        lstm_output, _ = self.lstm(bert_features)
        merged = torch.cat([bert_features, lstm_output], dim=-1)
        merged = torch.tanh(self.projection(merged))

        bool_mask = attention_mask.unsqueeze(-1).bool()
        max_masked = merged.masked_fill(~bool_mask, float("-inf"))
        max_pool = torch.max(max_masked, dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        float_mask = attention_mask.unsqueeze(-1).float()
        mean_pool = (merged * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp(min=1e-6)

        cls_pool = bert_features[:, 0, :]
        fused = torch.cat([cls_pool, max_pool, mean_pool], dim=-1)
        fused = torch.tanh(self.norm(self.fusion(self.dropout(fused))))
        return self.classifier(self.dropout(fused))


def build_classifier(model_type: str, num_labels: int, config: DeepTrainingConfig) -> nn.Module:
    model_name = model_type.lower()
    if model_name == "fc":
        return BertFCClassifier(
            pretrained_model_name=config.pretrained_model_name,
            num_labels=num_labels,
            dropout=config.dropout,
        )
    if model_name == "rcnn":
        return BertRCNNClassifier(
            pretrained_model_name=config.pretrained_model_name,
            num_labels=num_labels,
            hidden_size=config.rcnn_hidden_size,
            num_layers=config.rcnn_num_layers,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: np.ndarray, num_labels: int | None = None) -> np.ndarray:
    if labels.size == 0:
        return np.ones((num_labels or 1,), dtype=np.float32)

    upper = num_labels if num_labels is not None else (int(np.max(labels)) + 1)
    counts = np.bincount(labels.astype(int), minlength=upper).astype(np.float64)
    counts[counts <= 0] = 1.0
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def compute_sample_weights(labels: np.ndarray, class_weights: np.ndarray) -> np.ndarray:
    indices = labels.astype(int)
    return class_weights[indices]


def build_tokenizer_cache_key(source: str | Path, config: DeepTrainingConfig, *, extra: str = "") -> str:
    source_path = Path(source)
    if source_path.exists():
        stat = source_path.stat()
        source_meta = {
            "path": str(source_path.resolve()),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    else:
        source_meta = {"path": str(source)}
    payload = {
        "pretrained_model_name": config.pretrained_model_name,
        "max_length": int(config.max_length),
        "source": source_meta,
        "extra": extra,
    }
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:24]


def _resolve_loader_options(config: DeepTrainingConfig) -> dict[str, Any]:
    system_name = platform.system()
    pin_memory = torch.cuda.is_available() if config.pin_memory is None else bool(config.pin_memory)
    non_blocking = pin_memory if config.non_blocking is None else bool(config.non_blocking)
    num_workers = max(0, int(config.num_workers))

    if config.persistent_workers is None:
        persistent_workers = num_workers > 0 and system_name != "Windows"
    else:
        persistent_workers = bool(config.persistent_workers) and num_workers > 0

    prefetch_factor = None
    if num_workers > 0:
        if config.prefetch_factor is not None and int(config.prefetch_factor) > 0:
            prefetch_factor = int(config.prefetch_factor)
        else:
            prefetch_factor = 2

    return {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "non_blocking": non_blocking,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
    }


def _cache_path_from_key(cache_key: str, config: DeepTrainingConfig) -> Path:
    cache_root = Path(config.tokenizer_cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{cache_key}.pt"


def _batch_tokenize(
    texts: list[str],
    tokenizer: Any,
    max_length: int,
    *,
    batch_size: int = 512,
) -> dict[str, list[list[int]]]:
    encodings: dict[str, list[list[int]]] = {}
    for start in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", leave=False):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        for key, values in batch.items():
            encodings.setdefault(key, []).extend(values)
    return encodings


def _load_or_build_encodings(
    texts: list[str],
    tokenizer: Any,
    config: DeepTrainingConfig,
    cache_key: str | None,
) -> dict[str, list[list[int]]]:
    if not (config.enable_tokenizer_cache and cache_key):
        return _batch_tokenize(texts, tokenizer, config.max_length)

    cache_path = _cache_path_from_key(cache_key, config)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        if isinstance(payload, dict) and payload.get("num_texts") == len(texts):
            print(f"[Tokenizer cache] hit: {cache_path}")
            return payload["encodings"]

    encodings = _batch_tokenize(texts, tokenizer, config.max_length)
    payload = {
        "num_texts": len(texts),
        "encodings": encodings,
    }
    torch.save(payload, cache_path)
    print(f"[Tokenizer cache] built: {cache_path}")
    return encodings


class DeepChargeTrainer:
    def __init__(
        self,
        model_type: str,
        num_labels: int,
        config: DeepTrainingConfig,
        device: torch.device,
        class_weights: np.ndarray | torch.Tensor | None = None,
    ) -> None:
        self.model_type = model_type.lower()
        self.num_labels = num_labels
        self.config = config
        self.device = device
        self.model = build_classifier(self.model_type, num_labels, config)
        self.model.to(self.device)

        if class_weights is None:
            self.class_weights = None
        elif isinstance(class_weights, torch.Tensor):
            self.class_weights = class_weights.detach().clone().to(self.device)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)

        self.loss_fn = self._build_loss_fn()
        self.use_amp = bool(config.enable_amp) if config.enable_amp is not None else (self.device.type == "cuda")
        self.non_blocking = bool(config.non_blocking) if config.non_blocking is not None else (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _build_loss_fn(self) -> nn.Module:
        loss_name = (self.config.loss_name or "ce").lower()
        if loss_name == "focal":
            return FocalCrossEntropyLoss(
                gamma=float(self.config.focal_gamma),
                weight=self.class_weights,
                label_smoothing=float(self.config.label_smoothing),
            )
        weight = self.class_weights if loss_name == "weighted_ce" else None
        return nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=float(self.config.label_smoothing),
        )

    def _build_optimizer_and_scheduler(self, train_steps: int) -> tuple[torch.optim.Optimizer, Any]:
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        warmup_steps = int(train_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps,
        )
        return optimizer, scheduler

    def _autocast_context(self) -> Any:
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def _move_batch(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        labels = batch.pop("labels").to(self.device, non_blocking=self.non_blocking)
        inputs = {
            key: value.to(self.device, non_blocking=self.non_blocking)
            for key, value in batch.items()
        }
        return labels, inputs

    def _run_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> float:
        self.model.train()
        running_loss = 0.0
        accumulation = max(1, int(self.config.gradient_accumulation_steps))
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
            labels, inputs = self._move_batch(batch)
            with self._autocast_context():
                logits = self.model(**inputs)
                loss = self.loss_fn(logits, labels)

            running_loss += float(loss.detach().item())
            scaled_loss = loss / accumulation

            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % accumulation == 0 or step == len(dataloader):
                if self.use_amp:
                    self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if self.use_amp:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        return running_loss / max(len(dataloader), 1)

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            labels, inputs = self._move_batch(batch)
            with self._autocast_context():
                logits = self.model(**inputs)
                loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            total_loss += float(loss.item())

        metric_values = compute_classification_metrics(
            np.asarray(all_labels, dtype=int),
            np.asarray(all_preds, dtype=int),
        )
        return {
            "loss": total_loss / max(len(dataloader), 1),
            "accuracy": float(metric_values["accuracy"]),
            "f1_macro": float(metric_values["f1_macro"]),
            "f1_micro": float(metric_values["f1_micro"]),
            "f1_weighted": float(metric_values["f1_weighted"]),
        }

    @torch.no_grad()
    def collect_logits(self, dataloader: DataLoader) -> np.ndarray:
        self.model.eval()
        all_logits: list[np.ndarray] = []
        for batch in tqdm(dataloader, desc="Collecting logits", leave=False):
            batch.pop("labels")
            inputs = {
                key: value.to(self.device, non_blocking=self.non_blocking)
                for key, value in batch.items()
            }
            with self._autocast_context():
                logits = self.model(**inputs)
            all_logits.append(logits.detach().cpu().numpy())
        if not all_logits:
            return np.empty((0, self.num_labels), dtype=np.float32)
        return np.concatenate(all_logits, axis=0)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        output_dir: str | Path,
    ) -> tuple[dict[str, float], Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        accumulation = max(1, int(self.config.gradient_accumulation_steps))
        total_steps = math.ceil(max(len(train_loader), 1) / accumulation) * max(self.config.epochs, 1)
        optimizer, scheduler = self._build_optimizer_and_scheduler(total_steps)

        metric_name = self.config.selection_metric
        if metric_name not in {"accuracy", "f1_macro", "f1_micro", "f1_weighted"}:
            metric_name = "accuracy"

        best_metric_value = -1.0
        best_metrics: dict[str, float] = {}
        patience_count = 0
        best_model_path = output_path / f"best_{self.model_type}.pt"

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._run_epoch(train_loader, optimizer, scheduler)
            valid_metrics = self.evaluate(valid_loader)
            valid_metrics["train_loss"] = train_loss
            valid_metrics["epoch"] = float(epoch)

            current_value = float(valid_metrics.get(metric_name, 0.0))
            if current_value > best_metric_value:
                best_metric_value = current_value
                best_metrics = dict(valid_metrics)
                patience_count = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_count += 1

            if patience_count >= self.config.early_stopping_patience:
                break

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        return best_metrics, best_model_path


def build_dataloaders(
    train_texts: list[str],
    train_labels: np.ndarray,
    valid_texts: list[str],
    valid_labels: np.ndarray,
    test_texts: list[str],
    test_labels: np.ndarray,
    config: DeepTrainingConfig,
    *,
    train_cache_key: str | None = None,
    valid_cache_key: str | None = None,
    test_cache_key: str | None = None,
    train_sample_weights: np.ndarray | None = None,
    train_sampler_name: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Any]:
    tokenizer = load_tokenizer(config.pretrained_model_name)
    loader_options = _resolve_loader_options(config)

    if config.enable_tokenizer_cache:
        train_encodings = _load_or_build_encodings(train_texts, tokenizer, config, train_cache_key)
        valid_encodings = _load_or_build_encodings(valid_texts, tokenizer, config, valid_cache_key)
        test_encodings = _load_or_build_encodings(test_texts, tokenizer, config, test_cache_key)

        train_dataset: Dataset = EncodedTextDataset(train_encodings, train_labels)
        valid_dataset: Dataset = EncodedTextDataset(valid_encodings, valid_labels)
        test_dataset: Dataset = EncodedTextDataset(test_encodings, test_labels)
        collator: Any = EncodedBatchCollator(tokenizer)
    else:
        train_dataset = TextDataset(train_texts, train_labels)
        valid_dataset = TextDataset(valid_texts, valid_labels)
        test_dataset = TextDataset(test_texts, test_labels)
        collator = BatchTokenizerCollator(tokenizer, config.max_length)

    train_sampler = None
    train_sampler_name = (train_sampler_name or config.sampler_name or "none").lower()
    if train_sampler_name == "weighted" and train_sample_weights is not None and len(train_sample_weights) == len(train_dataset):
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(train_sample_weights, dtype=torch.double),
            num_samples=len(train_sample_weights),
            replacement=True,
        )

    train_kwargs = {
        "dataset": train_dataset,
        "batch_size": config.train_batch_size,
        "shuffle": train_sampler is None,
        "sampler": train_sampler,
        "num_workers": loader_options["num_workers"],
        "pin_memory": loader_options["pin_memory"],
        "collate_fn": collator,
    }
    if loader_options["num_workers"] > 0:
        train_kwargs["persistent_workers"] = loader_options["persistent_workers"]
        if loader_options["prefetch_factor"] is not None:
            train_kwargs["prefetch_factor"] = loader_options["prefetch_factor"]

    eval_common = {
        "batch_size": config.eval_batch_size,
        "shuffle": False,
        "num_workers": loader_options["num_workers"],
        "pin_memory": loader_options["pin_memory"],
        "collate_fn": collator,
    }
    if loader_options["num_workers"] > 0:
        eval_common["persistent_workers"] = loader_options["persistent_workers"]
        if loader_options["prefetch_factor"] is not None:
            eval_common["prefetch_factor"] = loader_options["prefetch_factor"]

    train_loader = DataLoader(**train_kwargs)
    valid_loader = DataLoader(valid_dataset, **eval_common)
    test_loader = DataLoader(test_dataset, **eval_common)
    return train_loader, valid_loader, test_loader, tokenizer


def build_predict_dataloader(
    texts: list[str],
    config: DeepTrainingConfig,
    tokenizer: Any | None = None,
) -> DataLoader:
    tokenizer = tokenizer or load_tokenizer(config.pretrained_model_name)
    loader_options = _resolve_loader_options(config)
    collator = BatchTokenizerCollator(tokenizer, config.max_length)
    dataset = TextDataset(texts, np.zeros(len(texts), dtype=np.int64))

    kwargs = {
        "dataset": dataset,
        "batch_size": config.eval_batch_size,
        "shuffle": False,
        "num_workers": loader_options["num_workers"],
        "pin_memory": loader_options["pin_memory"],
        "collate_fn": collator,
    }
    if loader_options["num_workers"] > 0:
        kwargs["persistent_workers"] = loader_options["persistent_workers"]
        if loader_options["prefetch_factor"] is not None:
            kwargs["prefetch_factor"] = loader_options["prefetch_factor"]
    return DataLoader(**kwargs)
