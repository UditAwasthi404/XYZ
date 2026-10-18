"""Draw detections and events: green products, red gaps, orange misplaced."""
from __future__ import annotations

import cv2
import numpy as np

GREEN, RED, ORANGE, WHITE, DARK = (60, 190, 60), (40, 40, 230), (0, 140, 255), (255, 255, 255), (30, 30, 30)


def _label(img, text, x, y, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)


def annotate(image: np.ndarray, products, gaps, events, title: str | None = None) -> np.ndarray:
    out = image.copy()
    misplaced_boxes = {tuple(e["bbox"]) for e in events if e["type"] == "MISPLACED"}
    for d in products:
        b = tuple(d.box())
        color = ORANGE if b in misplaced_boxes else GREEN
        cv2.rectangle(out, b[:2], b[2:], color, 3)
        _label(out, d.sku or "?", b[0], b[1] - 2, color)
    for g in gaps:
        b = g.box()
        overlay = out.copy()
        cv2.rectangle(overlay, tuple(b[:2]), tuple(b[2:]), RED, -1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.rectangle(out, tuple(b[:2]), tuple(b[2:]), RED, 3)
    for e in events:
        if e["type"] == "MISPLACED":
            continue
        b = e["bbox"]
        color = RED if e["type"] == "OUT_OF_STOCK" else ORANGE
        _label(out, f'{e["type"]} {e["sku"]}', int(b[0]), int(b[3]) + 16, color)
    if title:
        bar = np.full((44, out.shape[1], 3), DARK, np.uint8)
        cv2.putText(bar, title, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
        out = np.vstack([out, bar])  # footer keeps bbox coords aligned
    return out
