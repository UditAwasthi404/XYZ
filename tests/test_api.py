import importlib
import os
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


def test_audit_flow(tmp_path):
    os.environ["SHELFSENSE_DATA"] = str(tmp_path)
    import api.main as m
    importlib.reload(m)
    with TestClient(m.app) as c:
        def audit(name, shelf):
            with open(ROOT / "data/sample" / name, "rb") as f:
                r = c.post("/audit", files={"image": (name, f, "image/png")}, data={"shelf_id": shelf})
            assert r.status_code == 200, r.text
            return r.json()

        j = audit("A1-B1_t0.png", "A1-B1")
        assert len(j["created"]) == 3 and j["backend"] == "synthetic"
        assert c.get(j["evidence"]).status_code == 200
        eid = j["created"][0]["id"]
        assert c.post(f"/events/{eid}/ack").json()["status"] == "acked"
        assert len(c.get("/events?status=open").json()) == 3
        m1 = c.get("/metrics").json()
        assert m1["planogram_compliance_pct"] < 100
        j = audit("A1-B1_t1_restock.png", "A1-B1")
        assert len(j["auto_resolved"]) == 3 and j["still_open"] == 0
        m2 = c.get("/metrics").json()
        assert m2["planogram_compliance_pct"] == 100 and m2["mean_time_to_resolve_s"] is not None
        assert len(c.get("/events?status=resolved").json()) == 3
