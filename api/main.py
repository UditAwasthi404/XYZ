"""ShelfSense API.

    uvicorn api.main:app --reload
    open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from detector.pipeline import ShelfDetector
from planogram import load_planogram, reconcile
from planogram.auditor import run_audit

from .bus import EventBus
from .store import Store

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("SHELFSENSE_DATA", ROOT / "out"))
EVIDENCE_DIR = DATA_DIR / "evidence"
PLANOGRAM = Path(os.getenv("SHELFSENSE_PLANOGRAM", ROOT / "data" / "sample" / "planogram.json"))

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    state["store"] = Store(DATA_DIR / "shelfsense.db")
    state["planogram"] = load_planogram(PLANOGRAM)
    state["detector"] = ShelfDetector()
    state["bus"] = EventBus()
    await state["bus"].start()
    yield
    await state["bus"].stop()


app = FastAPI(title="ShelfSense", version="0.1.0", lifespan=lifespan,
              description="Real-time shelf auditing with vision AI (Everseen CV Hackathon, Team Void)")


def process_audit(image: np.ndarray, shelf_id: str, source_name: str, mode: str | None = None) -> tuple[dict, int]:
    store: Store = state["store"]
    det = state["detector"] if not mode or mode == "auto" else ShelfDetector(mode=mode)
    ev_name = f"{shelf_id}_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
    res = run_audit(image, shelf_id, state["planogram"], det, EVIDENCE_DIR / ev_name)
    evidence_url = f"/evidence/{ev_name}"
    res["evidence_url"] = evidence_url
    for e in res["events"]:
        e["evidence"] = evidence_url
    audit_id = store.add_audit(shelf_id, res["backend"], source_name, evidence_url,
                               len(res["products"]), len(res["gaps"]), res["slot_status"])
    return res, audit_id


@app.post("/audit")
async def audit(image: UploadFile = File(...), shelf_id: str = Form(...),
                detector: str = Form("auto", description="auto | synthetic | yolo")):
    buf = np.frombuffer(await image.read(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    try:
        res, audit_id = await asyncio.to_thread(process_audit, img, shelf_id, image.filename or "upload", detector)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    store: Store = state["store"]
    bus: EventBus = state["bus"]
    to_create, to_resolve = reconcile(store.open_events(shelf_id), res["events"], shelf_id)
    created = [store.create_event(e, audit_id) for e in to_create]
    resolved = [store.resolve(e["id"], audit_id) for e in to_resolve]
    for e in created:
        await bus.publish("created", e)
    for e in resolved:
        await bus.publish("resolved", e)
    return {
        "audit_id": audit_id, "shelf_id": shelf_id, "backend": res["backend"],
        "known_shelf": state["planogram"].bay(shelf_id) is not None,
        "products": len(res["products"]), "gaps": len(res["gaps"]), "people_blurred": res["people_blurred"],
        "events_detected": len(res["events"]), "created": created, "auto_resolved": resolved,
        "still_open": len(store.open_events(shelf_id)),
        "evidence": res["evidence_url"],
        "slot_status": res["slot_status"],
    }


@app.get("/events")
def events(status: str | None = Query(None, description="open | acked | resolved"), limit: int = 200):
    return state["store"].list_events(status, limit)


@app.post("/events/{eid}/ack")
async def ack(eid: int):
    store: Store = state["store"]
    if not store.get_event(eid):
        raise HTTPException(404, "event not found")
    e = store.ack(eid)
    await state["bus"].publish("acked", e)
    return e


@app.get("/metrics")
def metrics():
    store: Store = state["store"]
    slots = [s for a in store.latest_audits() for s in a["slot_status"]]
    n = len(slots)
    osa = 100 * sum(s["state"] != "OUT_OF_STOCK" for s in slots) / n if n else None
    compliance = 100 * sum(s["compliant"] for s in slots) / n if n else None
    dur = store.resolved_durations()
    c = store.counts()
    return {
        "on_shelf_availability_pct": round(osa, 1) if osa is not None else None,
        "planogram_compliance_pct": round(compliance, 1) if compliance is not None else None,
        "mean_time_to_resolve_s": round(sum(dur) / len(dur), 1) if dur else None,
        "slots_audited": n,
        "open_events": c.get("open", 0) + c.get("acked", 0),
        "acked_events": c.get("acked", 0),
        "resolved_events": c.get("resolved", 0),
    }


@app.get("/events/stream")
async def stream():
    """Server-Sent Events: one `data:` line per created/acked/resolved event."""
    bus: EventBus = state["bus"]
    q = bus.subscribe()

    async def gen():
        try:
            yield "retry: 2000\n\n"
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=15)
                    yield f"data: {json.dumps(msg)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            bus.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/audits/latest")
def latest_audits():
    """Most recent audit per bay (newest first)."""
    return sorted(state["store"].latest_audits(), key=lambda a: -a["id"])


@app.get("/shelves")
def shelves():
    return [{"shelf_id": b.shelf_id, "aisle": b.aisle} for b in state["planogram"].bays]


@app.get("/samples")
def samples():
    return json.loads((ROOT / "data" / "sample" / "scenarios.json").read_text())


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR), check_dir=False), name="evidence")
app.mount("/sample", StaticFiles(directory=str(ROOT / "data" / "sample")), name="sample")
app.mount("/static", StaticFiles(directory=str(ROOT / "dashboard")), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "dashboard" / "index.html")
