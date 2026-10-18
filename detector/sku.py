"""BASELINE SKU recognition: HSV colour-histogram matching.

Each product crop is compared against reference crops in data/catalog/
(one PNG per SKU, listed in catalog.json) with histogram correlation.
This is a deliberately simple, CPU-only, model-free baseline that works on
the synthetic demo. Production plan: replace with an embedding model
(e.g. a metric-learning CNN fine-tuned on catalogue pack shots) behind
the same `match()` interface.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import Detection

UNKNOWN = "UNKNOWN"


def _hist(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 60) & (hsv[..., 2] > 50)).astype(np.uint8)
    h = cv2.calcHist([hsv], [0, 1], mask, [36, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 1.0, 0, cv2.NORM_L1)
    return h


class ColorHistogramMatcher:
    def __init__(self, catalog_dir: str | Path, min_score: float = 0.5):
        self.catalog_dir = Path(catalog_dir)
        meta = json.loads((self.catalog_dir / "catalog.json").read_text())
        self.products = {p["sku"]: p for p in meta["products"]}
        self.refs: dict[str, np.ndarray] = {}
        for sku, p in self.products.items():
            img = cv2.imread(str(self.catalog_dir / p["reference"]))
            if img is not None:
                self.refs[sku] = _hist(img)
        self.min_score = min_score

    def match_crop(self, crop: np.ndarray) -> tuple[str, float]:
        if crop.size == 0 or not self.refs:
            return UNKNOWN, 0.0
        h = _hist(crop)
        best, score = UNKNOWN, -1.0
        for sku, ref in self.refs.items():
            s = float(cv2.compareHist(h, ref, cv2.HISTCMP_CORREL))
            if s > score:
                best, score = sku, s
        return (best if score >= self.min_score else UNKNOWN), max(score, 0.0)

    def assign(self, image: np.ndarray, dets: list[Detection]) -> list[Detection]:
        H, W = image.shape[:2]
        for d in dets:
            if d.cls != "product":
                continue
            x1, y1, x2, y2 = max(0, int(d.x1)), max(0, int(d.y1)), min(W, int(d.x2)), min(H, int(d.y2))
            d.sku, d.sku_score = self.match_crop(image[y1:y2, x1:x2])
        return dets
