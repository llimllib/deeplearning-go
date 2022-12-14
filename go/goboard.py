import copy
from go.gotypes import Point, Player, Move
from go.zobrist import EMPTY_BOARD, HASH_CODE
from go.scoring import compute_game_result
from typing import Dict, Iterable, List, Optional, Tuple


class GoString:
    def __init__(
        self, color: Player, stones: Iterable[Point], liberties: Iterable[Point]
    ):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def without_liberty(self, point: Point):
        return GoString(self.color, self.stones, self.liberties - set([point]))

    def with_liberty(self, point: Point):
        return GoString(self.color, self.stones, self.liberties | set([point]))

    # XXX: unclear why forward references aren't working here, so I had to string-type it
    def merged_with(self, go_string: "GoString") -> "GoString":
        assert go_string.color == self.color
        combined_stones = self.stones | go_string.stones
        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones,
        )

    @property
    def num_liberties(self) -> int:
        return len(self.liberties)

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, GoString)
            and self.color == other.color
            and self.stones == other.stones
            and self.liberties == other.liberties
        )


class Board:
    def __init__(self, num_rows: int, num_cols: int):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self._grid: Dict[Point, Optional[GoString]] = {}
        self._hash = EMPTY_BOARD

    def place_stone(self, player: Player, point: Point):
        assert self.is_on_grid(point)
        assert self._grid.get(point) is None

        adjacent_same_color: List[GoString] = []
        adjacent_opposite_color: List[GoString] = []

        liberties: List[Point] = []
        for neighbor in point.neighbors():
            if not self.is_on_grid(neighbor):
                continue
            neighbor_string = self._grid.get(neighbor)
            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)

        new_string = GoString(player, [point], liberties)

        # merge any adjacent strings of the same color
        for same_color_string in adjacent_same_color:
            new_string = new_string.merged_with(same_color_string)
        for new_string_point in new_string.stones:
            self._grid[new_string_point] = new_string

        # apply the hash code for this point and player to the zobrist hash
        self._hash ^= HASH_CODE[(point, player)]

        # replace liberties of any adjacent strings of the opposite color
        for other_color_string in adjacent_opposite_color:
            replacement = other_color_string.without_liberty(point)
            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            else:
                self._remove_string(other_color_string)

        # if any opposite-color strings now have zero liberties, remove them
        for other_color_string in adjacent_opposite_color:
            if other_color_string.num_liberties == 0:
                self._remove_string(other_color_string)

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols

    def get(self, point: Point) -> Optional[Player]:
        string = self._grid.get(point)
        if string is None:
            return None
        return string.color

    def get_go_string(self, point: Point) -> Optional[GoString]:
        string = self._grid.get(point)
        if string is None:
            return None
        return string

    def _replace_string(self, string: GoString):
        for point in string.stones:
            self._grid[point] = string

    def _remove_string(self, string: GoString):
        for point in string.stones:
            # removing a string can create liberties for other strings
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.with_liberty(point))
            self._grid[point] = None

            # unappply the hash for this move
            self._hash ^= HASH_CODE[point, string.color]

    def zobrist_hash(self) -> int:
        return self._hash


class GameState:
    def __init__(
        self,
        board: Board,
        next_player: Player,
        previous: Optional["GameState"],
        move: Optional[Move],
    ):
        self.board = board
        self.next_player = next_player
        self.previous_state = previous
        if previous is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states
                | {(previous.next_player, previous.board.zobrist_hash())}
            )
        self.last_move = move

    def apply_move(self, move: Move) -> "GameState":
        if move.is_play:
            next_board = copy.deepcopy(self.board)
            # since this is a play, point must not be None
            assert move.point
            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board
        return GameState(next_board, self.next_player.other, self, move)

    @classmethod
    def new_game(cls, board_size: int) -> "GameState":
        # the book's code allows int | Tuple[int, int] but tbh that's dumb and
        # we'll just allow one or the other
        board = Board(board_size, board_size)
        return GameState(board, Player.black, None, None)

    def is_over(self) -> bool:
        if self.last_move is None:
            return False
        if self.last_move.is_resign:
            return True
        assert self.previous_state
        second_last_move = self.previous_state.last_move
        if second_last_move is None:
            return False
        return self.last_move.is_pass and second_last_move.is_pass

    def is_move_self_capture(self, player: Player, move: Move) -> bool:
        if not move.is_play:
            return False

        # we have to do this every time we do move.is_play - could we solve
        # this with the type system?
        assert move.point

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        new_string = next_board.get_go_string(move.point)
        assert new_string
        return new_string.num_liberties == 0

    @property
    def situation(self) -> Tuple[Player, Board]:
        return (self.next_player, self.board)

    # section 3.5 updates this to use the zobrist hash, pg 51
    #
    # There's the author's fast version here:
    # https://github.com/maxpumperla/deep_learning_and_the_game_of_go/blob/35f983cabb7294d84f2554dc4c23063f23f985b8/code/dlgo/goboard_fast.py
    # I kind of want to write a benchmark and make mine faster than theirs
    # before I look at what they did?
    def does_move_violate_ko(self, player: Player, move: Move) -> bool:
        if not move.is_play:
            return False
        assert move.point

        next_board = copy.deepcopy(self.board)
        next_board.place_stone(player, move.point)
        next_situation = (player.other, next_board.zobrist_hash())
        return next_situation in self.previous_states

    def is_valid_move(self, move: Move) -> bool:
        if self.is_over():
            return False
        if move.is_pass or move.is_resign:
            return True
        assert move.point

        return (
            self.board.get(move.point) is None
            and not self.is_move_self_capture(self.next_player, move)
            and not self.does_move_violate_ko(self.next_player, move)
        )

    # had to pull this and `winner` from github because (AFAICT) they're not in
    # the book.
    #
    # https://github.com/maxpumperla/deep_learning_and_the_game_of_go/blob/6148f57eb98e4c75b102d096401efe780e911442/code/dlgo/goboard_slow.py
    def legal_moves(self) -> List[Move]:
        moves = []
        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play(Point(row, col))
                if self.is_valid_move(move):
                    moves.append(move)
        # These two moves are always legal.
        moves.append(Move.pass_turn())
        moves.append(Move.resign())

        return moves

    def winner(self) -> Optional[Player]:
        if not self.is_over() or not self.last_move:
            return None
        if self.last_move.is_resign:
            return self.next_player
        game_result = compute_game_result(self)
        return game_result.winner
