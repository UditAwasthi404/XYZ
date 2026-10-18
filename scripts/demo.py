"""End-to-end offline demo.

    python scripts/demo.py

1. Regenerates synthetic samples (deterministic).
2. Audits each bay at t0, prints events, saves annotated images to out/.
3. Audits the later restock images: open events are auto-closed.
4. Prints metrics and writes docs/demo.png for the README.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from api.store import Store  # noqa: E402
from detector.pipeline import ShelfDetector  # noqa: E402
from planogram import load_planogram, reconcile  # noqa: E402
from planogram.auditor import run_audit  # noqa: E402

OUT = ROOT / "out"
SEQUENCE = [  # (image, shelf_id) in time order
    ("A1-B1_t0.png", "A1-B1"),
    ("A1-B2_t0.png", "A1-B2"),
    ("A1-B2_t1_partial.png", "A1-B2"),
    ("A1-B1_t1_restock.png", "A1-B1"),
    ("A1-B2_t2_restock.png", "A1-B2"),
]
ICON = {"OUT_OF_STOCK": "[OOS ]", "LOW_STOCK": "[LOW ]", "MISPLACED": "[MISP]"}


def main():
    subprocess.run([sys.executable, str(ROOT / "data" / "generate_samples.py")], check=True)
    OUT.mkdir(exist_ok=True)
    db = OUT / "demo.db"
    db.unlink(missing_ok=True)
    store = Store(db)
    pg = load_planogram(ROOT / "data" / "sample" / "planogram.json")
    det = ShelfDetector(mode="auto")

    for i, (name, shelf_id) in enumerate(SEQUENCE):
        img = cv2.imread(str(ROOT / "data" / "sample" / name))
        ev_path = OUT / f"annotated_{name}"
        res = run_audit(img, shelf_id, pg, det, ev_path)
        audit_id = store.add_audit(shelf_id, res["backend"], name, str(ev_path),
                                   len(res["products"]), len(res["gaps"]), res["slot_status"])
        to_create, to_resolve = reconcile(store.open_events(shelf_id), res["events"], shelf_id)
        created = [store.create_event(e, audit_id) for e in to_create]
        resolved = [store.resolve(e["id"], audit_id) for e in to_resolve]
        print(f"\n=== Audit {i + 1}: {name} (bay {shelf_id}, detector={res['backend']}) ===")
        print(f"    {len(res['products'])} products, {len(res['gaps'])} gaps, {len(res['events'])} issues "
              f"-> {len(created)} new, {len(resolved)} auto-closed, "
              f"{len(res['events']) - len(created)} still open")
        for e in created:
            d = e["details"]
            extra = (f"{d['facings']}/{d['expected']} facings" if e["type"] == "LOW_STOCK"
                     else f"found {d['found_sku']} in {d['expected_sku']} slot" if e["type"] == "MISPLACED"
                     else "0 facings")
            print(f"    NEW  {ICON[e['type']]} {e['priority']:<6} aisle {e['aisle']} shelf {e['shelf']} "
                  f"{e['slot_id']:<12} {e['sku']:<9} {extra}  bbox={e['bbox']}")
        for e in resolved:
            print(f"    DONE {ICON[e['type']]} {e['slot_id']:<12} {e['sku']:<9} auto-closed by audit #{audit_id}")
        print(f"    annotated -> {ev_path.relative_to(ROOT)}")
        if name == "A1-B1_t0.png":
            (ROOT / "docs").mkdir(exist_ok=True)
            cv2.imwrite(str(ROOT / "docs" / "demo.png"), res["annotated"])
        time.sleep(0.2)  # so time-to-resolve is non-zero in the demo

    slots = [s for a in store.latest_audits() for s in a["slot_status"]]
    first = [s for s in store.db.execute("SELECT slot_status FROM audits WHERE id IN (1,2)")]
    import json
    t0_slots = [s for r in first for s in json.loads(r[0])]
    osa = lambda ss: 100 * sum(s["state"] != "OUT_OF_STOCK" for s in ss) / len(ss)  # noqa: E731
    comp = lambda ss: 100 * sum(s["compliant"] for s in ss) / len(ss)  # noqa: E731
    dur = store.resolved_durations()
    print("\n=== Metrics ===")
    print(f"    t0  : on-shelf availability {osa(t0_slots):.1f}%  planogram compliance {comp(t0_slots):.1f}%")
    print(f"    now : on-shelf availability {osa(slots):.1f}%  planogram compliance {comp(slots):.1f}%")
    print(f"    events: {store.counts()}  mean time to resolve {sum(dur) / len(dur):.2f}s (demo replays the day in seconds)")
    print(f"\nWrote annotated images to {OUT.relative_to(ROOT)}/ and docs/demo.png")
    open_left = store.open_events()
    assert not open_left, f"expected all events closed, {len(open_left)} still open"
    print("OK: every event was auto-closed by a later image.")


if __name__ == "__main__":
    main()
