"""SQLite persistence for audits and events (no ORM, stdlib only)."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audits (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  shelf_id TEXT, backend TEXT, image TEXT, evidence TEXT,
  n_products INTEGER, n_gaps INTEGER, slot_status TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  key TEXT, type TEXT, shelf_id TEXT, aisle TEXT, shelf INTEGER, slot_id TEXT, sku TEXT,
  priority TEXT, severity INTEGER, bbox TEXT, evidence TEXT, details TEXT,
  status TEXT DEFAULT 'open',            -- open | acked | resolved
  audit_id INTEGER, resolved_by_audit INTEGER,
  created_at REAL, acked_at REAL, resolved_at REAL
);
CREATE INDEX IF NOT EXISTS ix_events_status ON events(status, shelf_id);
"""


class Store:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    # -- audits ---------------------------------------------------------
    def add_audit(self, shelf_id, backend, image, evidence, n_products, n_gaps, slot_status) -> int:
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO audits(shelf_id, backend, image, evidence, n_products, n_gaps, slot_status, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (shelf_id, backend, image, evidence, n_products, n_gaps, json.dumps(slot_status), time.time()))
            return cur.lastrowid

    def latest_audits(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM audits WHERE id IN (SELECT MAX(id) FROM audits GROUP BY shelf_id)").fetchall()
        return [dict(r, slot_status=json.loads(r["slot_status"] or "[]")) for r in rows]

    # -- events ---------------------------------------------------------
    @staticmethod
    def _row(r) -> dict:
        d = dict(r)
        d["bbox"] = json.loads(d["bbox"])
        d["details"] = json.loads(d["details"] or "{}")
        return d

    def open_events(self, shelf_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM events WHERE status != 'resolved'", []
        if shelf_id:
            q += " AND shelf_id = ?"
            args.append(shelf_id)
        return [self._row(r) for r in self.db.execute(q, args)]

    def list_events(self, status: str | None = None, limit: int = 200) -> list[dict]:
        if status == "open":
            q, args = "SELECT * FROM events WHERE status != 'resolved'", []
        elif status:
            q, args = "SELECT * FROM events WHERE status = ?", [status]
        else:
            q, args = "SELECT * FROM events WHERE 1=1", []
        q += " ORDER BY (status='resolved'), severity DESC, created_at DESC LIMIT ?"
        return [self._row(r) for r in self.db.execute(q, args + [limit])]

    def get_event(self, eid: int) -> dict | None:
        r = self.db.execute("SELECT * FROM events WHERE id = ?", (eid,)).fetchone()
        return self._row(r) if r else None

    def create_event(self, e: dict, audit_id: int) -> dict:
        with self.lock, self.db:
            cur = self.db.execute(
                "INSERT INTO events(key,type,shelf_id,aisle,shelf,slot_id,sku,priority,severity,bbox,evidence,details,"
                "audit_id,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (e["key"], e["type"], e["shelf_id"], e["aisle"], e["shelf"], e["slot_id"], e["sku"], e["priority"],
                 e["severity"], json.dumps(e["bbox"]), e["evidence"], json.dumps(e.get("details", {})),
                 audit_id, time.time()))
        return self.get_event(cur.lastrowid)

    def ack(self, eid: int) -> dict | None:
        with self.lock, self.db:
            self.db.execute("UPDATE events SET status='acked', acked_at=? WHERE id=? AND status='open'",
                            (time.time(), eid))
        return self.get_event(eid)

    def resolve(self, eid: int, audit_id: int) -> dict | None:
        with self.lock, self.db:
            self.db.execute("UPDATE events SET status='resolved', resolved_at=?, resolved_by_audit=? "
                            "WHERE id=? AND status!='resolved'", (time.time(), audit_id, eid))
        return self.get_event(eid)

    def resolved_durations(self) -> list[float]:
        return [r[0] for r in self.db.execute(
            "SELECT resolved_at - created_at FROM events WHERE status='resolved'")]

    def counts(self) -> dict:
        return {r[0]: r[1] for r in self.db.execute("SELECT status, COUNT(*) FROM events GROUP BY status")}
