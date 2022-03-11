from dataclasses import dataclass

from .point import Point


@dataclass
class Rectangle:
    p1: Point
    p2: Point

    @staticmethod
    def new() -> 'Rectangle':
        return Rectangle(
            p1=Point.new(),
            p2=Point.new(),
        )
