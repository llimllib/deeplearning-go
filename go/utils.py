from go.gotypes import Player, Point
from go.goboard_slow import Board, Move

COLS = "ABCDEFGHJKLMNOPQRST"
STONE_TO_CHAR = {
    None: " . ",
    Player.black: " x ",
    Player.white: " o ",
}


def print_move(player: Player, move: Move):
    if move.is_pass:
        move_str = "passes"
    elif move.is_resign:
        move_str = "resigns"
    else:
        assert move.point
        move_str = f"{COLS[move.point.col - 1]}{move.point.row}"
    print(f"{player} {move_str}")


def print_board(board: Board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print(f"{bump}{row} {''.join(line)}")
    print(f"    {'  '.join(COLS[:board.num_cols])}")


def point_from_coords(coords: str) -> Point:
    col = COLS.index(coords[0].upper()) + 1
    row = int(coords[1:])
    return Point(row=row, col=col)
