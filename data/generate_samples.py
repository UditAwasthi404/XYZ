"""Generate deterministic synthetic shelf images + planogram + SKU catalogue.

    python data/generate_samples.py

Writes data/catalog/{catalog.json,*.png}, data/sample/planogram.json and
data/sample/*.png. Every scenario is listed in data/sample/scenarios.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
W, H = 1280, 720
BG, BOARD, EDGE = (225, 228, 232), (70, 78, 90), (50, 55, 62)
LEVEL_BOTTOMS = [230, 460, 690]           # y of each shelf board
PACK_W, PACK_H, PITCH = 70, 150, 90
SLOT_X0, SLOT_W = 40, 400

# sku, name, BGR colour, sales velocity (for prioritising)
CATALOG = [
    ("COLA-330", "Cola 330ml", (40, 40, 210), 5),
    ("ORNG-500", "Orange Juice 500ml", (20, 140, 245), 4),
    ("CHIP-050", "Masala Chips 50g", (20, 215, 235), 5),
    ("LIME-330", "Lime Soda 330ml", (60, 200, 70), 3),
    ("AQUA-1L", "Mineral Water 1L", (210, 120, 30), 4),
    ("GRAP-250", "Grape Drink 250ml", (160, 40, 140), 2),
]
COLOR = {s: c for s, _, c, _ in CATALOG}

LAYOUTS = {
    "A1-B1": [["COLA-330", "LIME-330", "AQUA-1L"], ["ORNG-500", "GRAP-250", "CHIP-050"], ["COLA-330", "ORNG-500", "AQUA-1L"]],
    "A1-B2": [["CHIP-050", "COLA-330", "GRAP-250"], ["AQUA-1L", "LIME-330", "ORNG-500"], ["CHIP-050", "AQUA-1L", "LIME-330"]],
}

# scenario name -> (shelf_id, {(level, slot): list of SKUs placed in facings 0..3, None = empty})
FULL = lambda sku: [sku] * 4  # noqa: E731
SCENARIOS = {
    "A1-B1_t0": ("A1-B1", {(0, 0): ["COLA-330", "AQUA-1L", "COLA-330", "COLA-330"],   # misplaced water
                           (0, 1): [None, None, None, None],                          # out of stock
                           (1, 2): ["CHIP-050", None, None, None],                    # low stock
                           (2, 1): ["ORNG-500", "ORNG-500", None, "ORNG-500"]}),      # one gap, still OK
    "A1-B1_t1_restock": ("A1-B1", {}),                                                # everything fixed
    "A1-B2_t0": ("A1-B2", {(1, 1): ["LIME-330", "LIME-330", None, None],              # low stock
                           (2, 0): ["CHIP-050", "CHIP-050", "GRAP-250", "CHIP-050"]}),# misplaced grape
    "A1-B2_t1_partial": ("A1-B2", {(2, 0): ["CHIP-050", "CHIP-050", "GRAP-250", "CHIP-050"],  # still wrong
                                   (0, 2): [None, None, None, None]}),                         # new OOS
    "A1-B2_t2_restock": ("A1-B2", {}),
}


def draw_pack(img, x, y_bottom, sku):
    x1, y1, x2, y2 = x, y_bottom - PACK_H, x + PACK_W, y_bottom
    cv2.rectangle(img, (x1, y1), (x2, y2), COLOR[sku], -1)
    cv2.rectangle(img, (x1 + 8, y1 + 55), (x2 - 8, y1 + 85), (245, 245, 245), -1)
    cv2.putText(img, sku.split("-")[0], (x1 + 11, y1 + 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c * 0.6) for c in COLOR[sku]), 2)


def render(shelf_id, overrides):
    img = np.full((H, W, 3), BG, np.uint8)
    for li, yb in enumerate(LEVEL_BOTTOMS):
        cv2.rectangle(img, (20, yb), (W - 20, yb + 18), BOARD, -1)
        cv2.line(img, (20, yb), (W - 20, yb), EDGE, 2)
        for si, sku in enumerate(LAYOUTS[shelf_id][li]):
            facings = overrides.get((li, si), FULL(sku))
            x0 = SLOT_X0 + si * SLOT_W
            for k, item in enumerate(facings):
                if item:
                    draw_pack(img, x0 + 25 + k * PITCH, yb - 2, item)
            cv2.rectangle(img, (x0 + 150, yb + 4), (x0 + 250, yb + 14), (205, 205, 205), -1)
    return img


def planogram():
    bays = []
    for shelf_id, levels in LAYOUTS.items():
        lv = []
        for li, skus in enumerate(levels):
            yb = LEVEL_BOTTOMS[li]
            lv.append({"shelf": li + 1, "y_range": [yb - PACK_H - 30, yb + 5], "slots": [
                {"slot_id": f"{shelf_id}-L{li + 1}-S{si + 1}", "sku": sku,
                 "x_range": [SLOT_X0 + si * SLOT_W, SLOT_X0 + (si + 1) * SLOT_W],
                 "expected_facings": 4, "low_stock_threshold": 2}
                for si, sku in enumerate(skus)]})
        bays.append({"shelf_id": shelf_id, "aisle": shelf_id.split("-")[0], "levels": lv})
    return {"store_id": "DEMO-001", "bays": bays}


def main():
    (ROOT / "catalog").mkdir(exist_ok=True)
    (ROOT / "sample").mkdir(exist_ok=True)
    prods = []
    for sku, name, _, vel in CATALOG:
        ref = np.full((PACK_H + 20, PACK_W + 20, 3), BG, np.uint8)
        draw_pack(ref, 10, PACK_H + 10, sku)
        cv2.imwrite(str(ROOT / "catalog" / f"{sku}.png"), ref[10:PACK_H + 10, 10:PACK_W + 10])
        prods.append({"sku": sku, "name": name, "reference": f"{sku}.png", "velocity": vel})
    (ROOT / "catalog" / "catalog.json").write_text(json.dumps({"products": prods}, indent=2))
    (ROOT / "sample" / "planogram.json").write_text(json.dumps(planogram(), indent=2))
    meta = []
    for name, (shelf_id, ov) in SCENARIOS.items():
        cv2.imwrite(str(ROOT / "sample" / f"{name}.png"), render(shelf_id, ov))
        meta.append({"image": f"{name}.png", "shelf_id": shelf_id})
    (ROOT / "sample" / "scenarios.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(meta)} sample images, planogram and {len(prods)} catalogue refs")


if __name__ == "__main__":
    main()
