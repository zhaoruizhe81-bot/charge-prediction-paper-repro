"""深度学习模型：BERT+FC / BERT+RCNN。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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


class DeepChargeTrainer:
    def __init__(
        self,
        model_type: str,
        num_labels: int,
        config: DeepTrainingConfig,
        device: torch.device,
    ) -> None:
        self.model_type = model_type.lower()
        self.num_labels = num_labels
        self.config = config
        self.device = device
        self.model = build_classifier(self.model_type, num_labels, config)
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()

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
            labels = batch.pop("labels").to(self.device)
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            logits = self.model(**inputs)
            loss = self.loss_fn(logits, labels)
            running_loss += loss.item()
            (loss / accumulation).backward()

            if step % accumulation == 0 or step == len(dataloader):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
            labels = batch.pop("labels").to(self.device)
            inputs = {key: value.to(self.device) for key, value in batch.items()}
            logits = self.model(**inputs)
            loss = self.loss_fn(logits, labels)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            total_loss += loss.item()

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
            inputs = {key: value.to(self.device) for key, value in batch.items()}
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
) -> tuple[DataLoader, DataLoader, DataLoader, Any]:
    tokenizer = load_tokenizer(config.pretrained_model_name)
    collator = BatchTokenizerCollator(tokenizer, config.max_length)

    train_dataset = TextDataset(train_texts, train_labels)
    valid_dataset = TextDataset(valid_texts, valid_labels)
    test_dataset = TextDataset(test_texts, test_labels)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collator,
    )
    return train_loader, valid_loader, test_loader, tokenizer


def build_predict_dataloader(
    texts: list[str],
    config: DeepTrainingConfig,
    tokenizer: Any | None = None,
) -> DataLoader:
    tokenizer = tokenizer or load_tokenizer(config.pretrained_model_name)
    collator = BatchTokenizerCollator(tokenizer, config.max_length)
    dataset = TextDataset(texts, np.zeros(len(texts), dtype=np.int64))
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )
