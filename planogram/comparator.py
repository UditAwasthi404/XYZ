"""Compare detections against the planogram and manage event lifecycle.

Events
  OUT_OF_STOCK  facings == 0                 priority HIGH   (score 3)
  LOW_STOCK     0 < facings <= threshold     priority MEDIUM (score 2)
  MISPLACED     a known SKU sits in another  priority LOW    (score 1)
                SKU's slot
Each event has a stable `key` (shelf_id/slot_id/type[/sku]) so that a later
audit of the same bay can auto-close it (reconcile()).
"""
from __future__ import annotations

from enum import Enum

from detector.sku import UNKNOWN
from detector.types import Detection

from .schema import Bay


class EventType(str, Enum):
    OUT_OF_STOCK = "OUT_OF_STOCK"
    LOW_STOCK = "LOW_STOCK"
    MISPLACED = "MISPLACED"


PRIORITY = {
    EventType.OUT_OF_STOCK: ("HIGH", 3),
    EventType.LOW_STOCK: ("MEDIUM", 2),
    EventType.MISPLACED: ("LOW", 1),
}


def _event(bay: Bay, level, slot, etype: EventType, sku: str, bbox, evidence, **details) -> dict:
    prio, score = PRIORITY[etype]
    key = f"{bay.shelf_id}/{slot.slot_id}/{etype.value}" + (f"/{sku}" if etype == EventType.MISPLACED else "")
    return {
        "key": key,
        "type": etype.value,
        "shelf_id": bay.shelf_id,
        "aisle": bay.aisle,
        "shelf": level.shelf,
        "slot_id": slot.slot_id,
        "sku": sku,
        "priority": prio,
        "severity": score,
        "bbox": [int(v) for v in bbox],
        "evidence": evidence,
        "details": details,
    }


def compare(bay: Bay, products: list[Detection], evidence: str | None = None) -> tuple[list[dict], list[dict]]:
    """Return (events, slot_status). slot_status has one row per planogram slot."""
    events: list[dict] = []
    status: list[dict] = []
    for level, slot in bay.slots():
        x1, x2 = slot.x_range
        y1, y2 = level.y_range
        inside = [d for d in products if x1 <= d.cx < x2 and y1 <= d.cy < y2]
        correct = [d for d in inside if d.sku == slot.sku]
        foreign = [d for d in inside if d.sku not in (slot.sku, UNKNOWN, None)]
        n = len(correct)
        region = (x1, y1, x2, y2)
        state = "OK"
        if n == 0:
            state = "OUT_OF_STOCK"
            events.append(_event(bay, level, slot, EventType.OUT_OF_STOCK, slot.sku, region, evidence,
                                 facings=0, expected=slot.expected_facings))
        elif n <= slot.low_stock_threshold:
            state = "LOW_STOCK"
            events.append(_event(bay, level, slot, EventType.LOW_STOCK, slot.sku, region, evidence,
                                 facings=n, expected=slot.expected_facings, threshold=slot.low_stock_threshold))
        for d in foreign:
            events.append(_event(bay, level, slot, EventType.MISPLACED, d.sku, d.box(), evidence,
                                 expected_sku=slot.sku, found_sku=d.sku))
        status.append({
            "slot_id": slot.slot_id, "sku": slot.sku, "shelf": level.shelf,
            "facings": n, "expected": slot.expected_facings, "misplaced": len(foreign),
            "state": state, "compliant": state == "OK" and not foreign,
        })
    events.sort(key=lambda e: -e["severity"])
    return events, status


def reconcile(open_events: list[dict], new_events: list[dict], shelf_id: str) -> tuple[list[dict], list[dict]]:
    """Given currently-open events and a fresh audit of `shelf_id`, return
    (to_create, to_resolve). An open event for this bay whose key is absent
    from the new audit is auto-closed: the later image proves the fix."""
    new_keys = {e["key"] for e in new_events}
    open_keys = {e["key"] for e in open_events if e["shelf_id"] == shelf_id}
    to_create = [e for e in new_events if e["key"] not in open_keys]
    to_resolve = [e for e in open_events if e["shelf_id"] == shelf_id and e["key"] not in new_keys]
    return to_create, to_resolve
