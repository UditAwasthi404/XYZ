"""Digital planogram schema.

{
  "store_id": "DEMO-001",
  "bays": [{
    "shelf_id": "A1-B1",        # camera / bay id used by POST /audit
    "aisle": "A1",
    "levels": [{
      "shelf": 1,               # shelf level (1 = top)
      "y_range": [60, 230],     # pixel band of this shelf in the camera frame
      "slots": [{
        "slot_id": "A1-B1-L1-S1",
        "sku": "COLA-330",
        "x_range": [40, 440],
        "expected_facings": 4,
        "low_stock_threshold": 2   # LOW_STOCK when 0 < facings <= threshold
      }]
    }]
  }]
}
Pixel ranges assume a fixed camera per bay (calibrated once at install).
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class Slot(BaseModel):
    slot_id: str
    sku: str
    x_range: tuple[float, float]
    expected_facings: int = Field(ge=1)
    low_stock_threshold: int = Field(ge=0)


class Level(BaseModel):
    shelf: int
    y_range: tuple[float, float]
    slots: list[Slot]


class Bay(BaseModel):
    shelf_id: str
    aisle: str
    levels: list[Level]

    def slots(self):
        for lv in self.levels:
            for s in lv.slots:
                yield lv, s

    def row_bounds(self) -> list[tuple[float, float, float, float]]:
        out = []
        for lv in self.levels:
            xs = [x for s in lv.slots for x in s.x_range]
            out.append((min(xs), lv.y_range[0], max(xs), lv.y_range[1]))
        return out


class Planogram(BaseModel):
    store_id: str
    bays: list[Bay]

    def bay(self, shelf_id: str) -> Bay | None:
        return next((b for b in self.bays if b.shelf_id == shelf_id), None)


def load_planogram(path: str | Path) -> Planogram:
    return Planogram.model_validate(json.loads(Path(path).read_text()))
