"""ShelfSense detection package: product/gap detection and baseline SKU matching."""
from .types import Detection
from .pipeline import ShelfDetector

__all__ = ["Detection", "ShelfDetector"]
