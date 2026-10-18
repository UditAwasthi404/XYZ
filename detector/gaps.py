"""Heuristic empty-facing (gap) detection.

1. Cluster product boxes into shelf rows by their vertical centre.
2. Within each row, sort by x and flag horizontal spaces wider than
   `factor` x the median product width as gaps (including the space between
   the shelf edges and the first/last product when row bounds are known).
"""
from __future__ import annotations

from statistics import median

from .types import Detection


def cluster_rows(products: list[Detection], tol: float = 0.5) -> list[list[Detection]]:
    if not products:
        return []
    mh = median(d.h for d in products)
    rows: list[list[Detection]] = []
    for d in sorted(products, key=lambda d: d.cy):
        if rows and abs(d.cy - sum(p.cy for p in rows[-1]) / len(rows[-1])) <= tol * mh:
            rows[-1].append(d)
        else:
            rows.append([d])
    return [sorted(r, key=lambda d: d.x1) for r in rows]


def find_gaps(
    products: list[Detection],
    row_bounds: list[tuple[float, float, float, float]] | None = None,
    factor: float = 1.2,
) -> list[Detection]:
    """row_bounds: optional list of (x1, y1, x2, y2) shelf regions (from the planogram).

    With bounds, gaps are searched inside each region (so an entirely empty
    shelf row still yields a gap). Without bounds, rows come from clustering.
    """
    if not products and not row_bounds:
        return []
    mw = median(d.w for d in products) if products else 70.0
    gaps: list[Detection] = []

    def scan(row: list[Detection], x_start: float, x_end: float, y1: float, y2: float):
        cursor = x_start
        for d in row:
            if d.x1 - cursor > factor * mw:
                gaps.append(Detection(cursor, y1, d.x1, y2, cls="gap", conf=0.9))
            cursor = max(cursor, d.x2)
        if x_end - cursor > factor * mw:
            gaps.append(Detection(cursor, y1, x_end, y2, cls="gap", conf=0.9))

    if row_bounds:
        for (bx1, by1, bx2, by2) in row_bounds:
            row = sorted([d for d in products if bx1 <= d.cx <= bx2 and by1 <= d.cy <= by2], key=lambda d: d.x1)
            ys = (min(d.y1 for d in row), max(d.y2 for d in row)) if row else (by1, by2)
            scan(row, bx1, bx2, *ys)
    else:
        for row in cluster_rows(products):
            y1, y2 = min(d.y1 for d in row), max(d.y2 for d in row)
            scan(row, row[0].x1, row[-1].x2, y1, y2)  # interior gaps only
    return gaps
