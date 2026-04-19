"""罪名预测训练工具包。"""

from __future__ import annotations

from typing import Any


def compute_classification_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Lazy import to keep lightweight submodules usable without sklearn runtime."""
    from .metrics import compute_classification_metrics as _impl

    return _impl(*args, **kwargs)


def compute_multilabel_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Lazy import for law-article multi-label evaluation."""
    from .metrics import compute_multilabel_metrics as _impl

    return _impl(*args, **kwargs)


__all__ = ["compute_classification_metrics", "compute_multilabel_metrics"]
