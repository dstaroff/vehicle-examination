import colorsys
from typing import (
    NewType,
    Optional,
)

import numpy as np

from src.gui.model import Color
from src.gui.model.geometry import Point
from src.gui.model.geometry import Rectangle
from src.utils import const

_hsv_colors = [(i / 2, 1, 1.0) for i in range(2)]
_gbr_colors = []
for h, s, v in _hsv_colors:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    _gbr_colors.append(Color((
        round(g * 255),
        round(b * 255),
        round(r * 255),
    )))


class Subject:
    _type_id: int
    _caption: str

    _mask: np.ndarray

    def __init__(self):
        self._color = _gbr_colors[self._type_id]

        self._bounding_box: Rectangle = Rectangle.new()
        self._centroid: Point = Point.new()
        self._score: float = 0.0

    def caption(self) -> str:
        return self._caption

    def color(self) -> Color:
        return self._color

    def set_color(self, color: Color):
        self._color = color

    def centroid(self) -> Point:
        return self._centroid

    def set_centroid(self, point: Point):
        self._centroid = point

    def score(self) -> float:
        return self._score

    def set_score(self, score: float):
        self._score = score

    def mask(self) -> np.ndarray:
        return self._mask

    def set_mask(self, mask: np.ndarray):
        self._mask = mask

    def mask_area(self) -> int:
        return np.count_nonzero(self._mask)

    def bounding_box(self) -> Rectangle:
        return self._bounding_box

    def set_bounding_box(self, bounding_box: Rectangle):
        self._bounding_box = bounding_box


SubjectImpl = NewType('SubjectImpl', Subject)


class Person(Subject):
    _type_id = 0
    _caption = 'Examiner'


class Vehicle(Subject):
    _type_id = 1
    _caption = 'Vehicle'


def subject_from_class_id(class_id: int) -> Optional[SubjectImpl]:
    class_name = const.CLASS_NAMES_ALL[class_id]

    if class_name == 'person':
        return Person()
    elif class_name in ['car', 'motorcycle', 'truck', 'bus']:
        return Vehicle()

    return None
