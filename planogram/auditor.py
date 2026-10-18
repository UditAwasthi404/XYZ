"""One-call audit: image -> detections -> planogram events -> annotated evidence."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detector.annotate import annotate
from detector.pipeline import ShelfDetector

from .comparator import compare
from .schema import Planogram


def gap_events(shelf_id: str, gaps, evidence) -> list[dict]:
    """No planogram for this camera: every detected gap becomes an OOS task."""
    evs = []
    for g in gaps:
        b = g.box()
        bucket = f"x{b[0] // 100}y{b[1] // 100}"
        evs.append({"key": f"{shelf_id}/gap-{bucket}/OUT_OF_STOCK", "type": "OUT_OF_STOCK",
                    "shelf_id": shelf_id, "aisle": shelf_id.split("-")[0], "shelf": None,
                    "slot_id": f"gap-{bucket}", "sku": "UNKNOWN", "priority": "HIGH", "severity": 3,
                    "bbox": b, "evidence": evidence, "details": {"source": "gap-heuristic"}})
    return evs


def run_audit(image: np.ndarray, shelf_id: str, planogram: Planogram, detector: ShelfDetector,
              evidence_path: str | Path | None = None) -> dict:
    bay = planogram.bay(shelf_id)
    det = detector.run(image, row_bounds=bay.row_bounds() if bay else None)
    ev = str(evidence_path) if evidence_path else None
    if bay:
        events, status = compare(bay, det["products"], ev)
    else:
        events, status = gap_events(shelf_id, det["gaps"], ev), []
    annotated = annotate(det["image"], det["products"], det["gaps"], events,
                         title=f"ShelfSense | {shelf_id} | {len(events)} events")
    if evidence_path:
        Path(evidence_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(evidence_path), annotated)  # person-blurred before annotation
    return {"shelf_id": shelf_id, "backend": det["backend"], "events": events, "slot_status": status,
            "products": [d.to_dict() for d in det["products"]], "gaps": [g.to_dict() for g in det["gaps"]],
            "people_blurred": det["people_blurred"], "annotated": annotated}
