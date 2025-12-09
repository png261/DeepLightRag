"""
Geometry utilities for OCR processing
"""

from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class BoundingBox:
    """Bounding box for visual region"""

    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        return cls(coords[0], coords[1], coords[2], coords[3])

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def height(self) -> float:
        return self.y2 - self.y1

    def width(self) -> float:
        return self.x2 - self.x1
