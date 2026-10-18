"""Privacy filter: blur every COCO 'person' box before an image is stored.

ShelfSense never stores faces or person crops: person detections are used
only to blur the region, then discarded (never written to the DB).
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import Detection


def blur_people(image: np.ndarray, dets: list[Detection], pad: int = 10) -> tuple[np.ndarray, int]:
    out = image.copy()
    H, W = out.shape[:2]
    n = 0
    for d in dets:
        if d.cls != "person":
            continue
        x1, y1 = max(0, int(d.x1) - pad), max(0, int(d.y1) - pad)
        x2, y2 = min(W, int(d.x2) + pad), min(H, int(d.y2) + pad)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = out[y1:y2, x1:x2]
        k = max(31, (min(roi.shape[:2]) // 3) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        n += 1
    return out, n
