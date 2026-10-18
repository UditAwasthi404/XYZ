import cv2
import pytest

from detector.gaps import find_gaps
from detector.pipeline import ShelfDetector
from detector.types import Detection
from planogram import Bay, compare, reconcile
from planogram.schema import load_planogram

ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent

BAY = Bay.model_validate({
    "shelf_id": "T-1", "aisle": "T",
    "levels": [{"shelf": 1, "y_range": [0, 200], "slots": [
        {"slot_id": "S1", "sku": "COLA", "x_range": [0, 400], "expected_facings": 4, "low_stock_threshold": 2},
        {"slot_id": "S2", "sku": "LIME", "x_range": [400, 800], "expected_facings": 4, "low_stock_threshold": 2},
    ]}],
})


def pack(x, sku):
    return Detection(x, 40, x + 70, 190, sku=sku)


def test_all_ok_no_events():
    prods = [pack(25 + 90 * k, "COLA") for k in range(4)] + [pack(425 + 90 * k, "LIME") for k in range(4)]
    events, status = compare(BAY, prods)
    assert events == []
    assert all(s["compliant"] for s in status)


def test_out_of_stock_low_stock_and_misplaced():
    prods = [pack(25, "COLA"), pack(115, "LIME"), pack(205, "COLA"), pack(295, "COLA")]  # S2 empty
    events, status = compare(BAY, prods, evidence="ev.jpg")
    types = {(e["type"], e["slot_id"], e["sku"]) for e in events}
    assert ("OUT_OF_STOCK", "S2", "LIME") in types
    assert ("MISPLACED", "S1", "LIME") in types
    assert events[0]["type"] == "OUT_OF_STOCK" and events[0]["priority"] == "HIGH"  # sorted by severity
    mis = next(e for e in events if e["type"] == "MISPLACED")
    assert mis["bbox"] == [115, 40, 185, 190] and mis["evidence"] == "ev.jpg"
    assert all(k in mis for k in ("aisle", "shelf", "sku", "severity", "bbox", "evidence"))

    events, _ = compare(BAY, [pack(25, "COLA"), pack(425, "LIME"), pack(515, "LIME"), pack(605, "LIME")])
    low = [e for e in events if e["type"] == "LOW_STOCK"]
    assert len(low) == 1 and low[0]["sku"] == "COLA" and low[0]["details"]["facings"] == 1


def test_threshold_boundary():
    three = [pack(25 + 90 * k, "COLA") for k in range(3)] + [pack(425 + 90 * k, "LIME") for k in range(4)]
    assert compare(BAY, three)[0] == []  # 3 > threshold 2 -> fine
    two = three[1:]
    assert [e["type"] for e in compare(BAY, two)[0]] == ["LOW_STOCK"]


def test_unknown_sku_is_not_misplaced():
    prods = [pack(25 + 90 * k, "COLA") for k in range(4)] + [pack(425 + 90 * k, "LIME") for k in range(3)]
    prods.append(pack(695, "UNKNOWN"))
    assert compare(BAY, prods)[0] == []


def test_reconcile_auto_close():
    bad, _ = compare(BAY, [pack(25, "COLA")])
    open_events = [dict(e, id=i) for i, e in enumerate(bad)]
    create, resolve = reconcile([], bad, "T-1")
    assert len(create) == len(bad) and resolve == []
    # same problems seen again -> nothing new, nothing closed
    create, resolve = reconcile(open_events, bad, "T-1")
    assert create == [] and resolve == []
    # later image shows the slot fixed -> all auto-close
    create, resolve = reconcile(open_events, [], "T-1")
    assert create == [] and len(resolve) == len(open_events)
    # an audit of a different bay never closes these
    assert reconcile(open_events, [], "OTHER")[1] == []


def test_gap_heuristic():
    prods = [pack(25, "COLA"), pack(115, "COLA"), pack(385, "COLA")]  # hole between 185 and 385
    gaps = find_gaps(prods)
    assert len(gaps) == 1 and gaps[0].x1 == 185 and gaps[0].x2 == 385
    # with row bounds an entirely empty row is one gap
    assert len(find_gaps(prods, [(0, 0, 800, 200), (0, 200, 800, 400)])) == 3


@pytest.mark.parametrize("name,shelf,expected", [
    ("A1-B1_t0.png", "A1-B1", {"OUT_OF_STOCK", "LOW_STOCK", "MISPLACED"}),
    ("A1-B1_t1_restock.png", "A1-B1", set()),
])
def test_synthetic_end_to_end(name, shelf, expected):
    pg = load_planogram(ROOT / "data/sample/planogram.json")
    det = ShelfDetector(mode="auto")
    img = cv2.imread(str(ROOT / "data/sample" / name))
    out = det.run(img, pg.bay(shelf).row_bounds())
    assert out["backend"] == "synthetic"
    events, _ = compare(pg.bay(shelf), out["products"])
    assert {e["type"] for e in events} == expected
