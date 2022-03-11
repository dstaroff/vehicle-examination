from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float

    @staticmethod
    def new() -> 'Point':
        return Point(
            x=0,
            y=0,
        )
