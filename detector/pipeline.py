"""ShelfDetector: picks a backend, runs detection, gaps, SKU matching, privacy blur."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .gaps import find_gaps
from .privacy import blur_people
from .sku import ColorHistogramMatcher
from .synthetic import SyntheticDetector, looks_synthetic
from .types import Detection

ROOT = Path(__file__).resolve().parent.parent


class ShelfDetector:
    """mode: 'auto' (synthetic for generated images, YOLO otherwise), 'synthetic', 'yolo'."""

    def __init__(self, mode: str | None = None, catalog_dir: str | Path | None = None, weights: str | None = None):
        self.mode = mode or os.getenv("SHELFSENSE_DETECTOR", "auto")
        self.weights = weights
        self._synthetic = SyntheticDetector()
        self._yolo = None
        self.matcher = ColorHistogramMatcher(catalog_dir or ROOT / "data" / "catalog")

    def _get_yolo(self):
        if self._yolo is None:
            from .yolo import YoloDetector
            self._yolo = YoloDetector(self.weights)
        return self._yolo

    def pick_backend(self, image: np.ndarray) -> str:
        if self.mode in ("synthetic", "yolo"):
            return self.mode
        if looks_synthetic(image):
            return "synthetic"
        try:
            import ultralytics  # noqa: F401
            return "yolo"
        except ImportError:
            return "synthetic"

    def run(self, image: np.ndarray, row_bounds=None) -> dict:
        backend = self.pick_backend(image)
        raw = self._get_yolo().detect(image) if backend == "yolo" else self._synthetic.detect(image)
        clean, blurred = blur_people(image, raw)            # privacy first
        products = [d for d in raw if d.cls == "product"]
        model_gaps = [d for d in raw if d.cls == "gap"]
        self.matcher.assign(clean, products)
        gaps = model_gaps + find_gaps(products, row_bounds)
        return {
            "backend": backend,
            "image": clean,                                  # person-blurred image, safe to store
            "products": products,
            "gaps": gaps,
            "people_blurred": blurred,                       # count only; boxes are discarded
        }


__all__ = ["ShelfDetector", "Detection"]
