from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Detection:
    """One box in image pixel coordinates (x1, y1, x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float
    cls: str = "product"          # product | gap | price_tag | person
    conf: float = 1.0
    sku: Optional[str] = None      # filled by the SKU matcher for products
    sku_score: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    def box(self) -> list[int]:
        return [int(self.x1), int(self.y1), int(self.x2), int(self.y2)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["box"] = self.box()
        return d
