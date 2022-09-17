import enum
from collections import namedtuple
from typing import List, Optional


class Player(enum.Enum):
    black = 1
    white = 2

    @property
    def other(self) -> "Player":
        return Player.black if self == Player.white else Player.white


class Point(namedtuple("Point", "row col")):
    def neighbors(self) -> List["Point"]:
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row, self.col - 1),
            Point(self.row, self.col + 1),
        ]


class Move:
    def __init__(
        self,
        point: Optional[Point] = None,
        is_pass: bool = False,
        is_resign: bool = False,
    ):
        assert (point is not None) ^ is_pass ^ is_resign
        self.point = point
        self.is_play = self.point is not None
        self.is_pass = is_pass
        self.is_resign = is_resign

    @classmethod
    def play(cls, point: Point) -> "Move":
        return Move(point=point)

    @classmethod
    def pass_turn(cls) -> "Move":
        return Move(is_pass=True)

    @classmethod
    def resign(cls) -> "Move":
        return Move(is_resign=True)
