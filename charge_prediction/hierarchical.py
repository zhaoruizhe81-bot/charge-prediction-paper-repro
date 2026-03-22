"""层次分类器：先大类，再细粒度罪名。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

from .ml_models import build_model


@dataclass
class CategoryModel:
    model: Any
    labels: list[int]


class HierarchicalChargeClassifier:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.coarse_model = build_model("svm", seed=seed)
        self.global_fine_model = build_model("svm", seed=seed)
        self.category_models: dict[int, CategoryModel] = {}
        self._is_fitted = False

    def fit(
        self,
        texts: list[str],
        coarse_labels: np.ndarray,
        fine_labels: np.ndarray,
    ) -> "HierarchicalChargeClassifier":
        unique_categories = sorted(set(int(item) for item in coarse_labels))
        if len(unique_categories) <= 1:
            coarse_dummy = DummyClassifier(strategy="constant", constant=unique_categories[0])
            coarse_dummy.fit(texts, coarse_labels)
            self.coarse_model = coarse_dummy
        else:
            self.coarse_model.fit(texts, coarse_labels)

        self.global_fine_model.fit(texts, fine_labels)

        for category_id in unique_categories:
            indices = np.where(coarse_labels == category_id)[0]
            category_texts = [texts[index] for index in indices]
            category_fine = fine_labels[indices]
            fine_unique = sorted(set(int(item) for item in category_fine))

            if len(fine_unique) <= 1:
                dummy = DummyClassifier(strategy="constant", constant=fine_unique[0])
                dummy.fit(category_texts, category_fine)
                self.category_models[category_id] = CategoryModel(model=dummy, labels=fine_unique)
                continue

            model = clone(self.global_fine_model)
            model.fit(category_texts, category_fine)
            self.category_models[category_id] = CategoryModel(model=model, labels=fine_unique)

        self._is_fitted = True
        return self

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted.")

        coarse_pred = self.coarse_model.predict(texts)
        global_pred = self.global_fine_model.predict(texts)
        fine_pred = np.array(global_pred, copy=True)

        for category_id, category_model in self.category_models.items():
            indices = np.where(coarse_pred == category_id)[0]
            if len(indices) == 0:
                continue
            subset_texts = [texts[index] for index in indices]
            subset_pred = category_model.model.predict(subset_texts)
            fine_pred[indices] = subset_pred

        return coarse_pred, fine_pred
