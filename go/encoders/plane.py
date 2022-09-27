from typing import Tuple

import numpy as np
import numpy.typing as nptype

from go.encoders.encoder import Encoder
from go.goboard import Point, GameState


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size: int):
        # the book lets you encode non-square boards, I'm just not
        # allowing that
        self.board_width = self.board_height = board_size
        self.num_planes = 1

    def name(self) -> str:
        return "oneplane"

    def encode(self, game_state: GameState) -> nptype.NDArray[np.float64]:
        board_matrix = np.zeros(self.shape())
        for r in range(self.board_height):
            for c in range(self.board_width):
                p = Point(row=r + 1, col=c + 1)
                go_string = game_state.board.get_go_string(p)
                if go_string is None:
                    continue
                if go_string.color == game_state.next_player:
                    board_matrix[0, r, c] = 1
                else:
                    board_matrix[0, r, c] = -1

        return board_matrix

    def encode_point(self, point: Point) -> int:
        return self.board_width * (point.row - 1) + (point.col - 1)

    def decode_point_index(self, index: int) -> Point:
        row = index // self.board_width
        col = index % self.board_width
        return Point(row=row + 1, col=col + 1)

    def num_points(self) -> int:
        return self.board_width * self.board_height

    def shape(self) -> Tuple[int, int, int]:
        return self.num_planes, self.board_height, self.board_width
