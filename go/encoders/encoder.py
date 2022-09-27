from abc import ABC, abstractmethod
from importlib import import_module
from typing import Tuple

import numpy as np
import numpy.typing as nptype

from go.goboard import GameState
from go.gotypes import Point


class Encoder(ABC):
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def encode(self, game_state: GameState) -> nptype.NDArray[np.float64]:
        raise NotImplementedError()

    @abstractmethod
    def encode_point(self, point: Point) -> int:
        """Turn a board point into an index"""
        raise NotImplementedError()

    @abstractmethod
    def decode_point_index(self, index: int) -> Point:
        """Turn a board index into a Point"""
        raise NotImplementedError()

    @abstractmethod
    def num_points(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """return (# of planes, height, width)"""
        raise NotImplementedError()


# XXX: I'm v suspicious of this function, can we eliminate it? pg. 120
def get_encoder_by_name(name: str, board_size: int):
    module = import_module(f"go.encoders.{name}")
    constructor = getattr(module, "create")
    return constructor((board_size, board_size))
