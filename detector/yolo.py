"""Ultralytics YOLOv8 wrapper (lazy import so the demo runs without torch).

* Default weights: yolov8n.pt (COCO-pretrained). Every non-person box is
  treated as a product; COCO 'person' boxes are returned separately so the
  privacy filter can blur them before any evidence image is stored.
* Fine-tuned weights (see detector/train.py, SKU-110K): class 0 'object' ->
  product, optional class 'gap' -> empty facing, optional 'price_tag'.
"""
from __future__ import annotations

import os

import numpy as np

from .types import Detection

PERSON = "person"


class YoloDetector:
    name = "yolo"

    def __init__(self, weights: str | None = None, conf: float = 0.25, imgsz: int = 1024):
        try:
            from ultralytics import YOLO  # noqa: WPS433 (lazy on purpose)
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed. `pip install ultralytics` or use detector mode 'synthetic'."
            ) from e
        self.weights = weights or os.getenv("SHELFSENSE_WEIGHTS", "yolov8n.pt")
        self.model = YOLO(self.weights)
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image: np.ndarray) -> list[Detection]:
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False, max_det=1000)[0]
        names = res.names
        H, W = image.shape[:2]
        out: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            label = names[int(b.cls)]
            if label != PERSON and (x2 - x1) * (y2 - y1) > 0.2 * W * H:
                continue  # COCO weights: huge non-person boxes (bus, fridge...) are scene, not a facing
            if label == PERSON:
                cls = "person"
            elif label in ("gap", "empty"):
                cls = "gap"
            elif label in ("price_tag", "price-tag", "label"):
                cls = "price_tag"
            else:
                cls = "product"
            out.append(Detection(x1, y1, x2, y2, cls=cls, conf=float(b.conf), extra={"label": label}))
        return out
