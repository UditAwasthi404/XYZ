"""Synthetic detector: finds saturated coloured rectangles with OpenCV.

Used for the generated sample images in data/sample/ so the demo is fully
offline and deterministic. Real shelf photos go through the YOLO detector.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


class SyntheticDetector:
    name = "synthetic"

    def __init__(self, min_area: int = 1500, sat_thresh: int = 80, val_thresh: int = 60):
        self.min_area = min_area
        self.sat_thresh = sat_thresh
        self.val_thresh = val_thresh

    def detect(self, image: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = ((hsv[..., 1] > self.sat_thresh) & (hsv[..., 2] > self.val_thresh)).astype(np.uint8) * 255
        # close small holes (printed label text inside each pack)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dets = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < self.min_area or h < w * 0.8:  # products are upright packs
                continue
            dets.append(Detection(x, y, x + w, y + h, cls="product", conf=0.99))
        dets.sort(key=lambda d: (round(d.cy / 50), d.x1))
        return dets


def looks_synthetic(image: np.ndarray, top_k: int = 16, frac: float = 0.85) -> bool:
    """Heuristic: generated images are made of a handful of flat colours."""
    small = cv2.resize(image, (160, 90), interpolation=cv2.INTER_NEAREST)
    q = (small // 8).reshape(-1, 3)
    keys = q[:, 0].astype(np.int32) * 1024 + q[:, 1].astype(np.int32) * 32 + q[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    counts.sort()
    return counts[-top_k:].sum() / keys.size >= frac
