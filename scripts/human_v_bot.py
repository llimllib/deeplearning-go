#!/usr/bin/env python
import os
import sys

sys.path.insert(0, os.path.dirname(sys.argv[0]) + "/..")

from go.goboard import GameState
from go.agent.naive import RandomBot
from go.gotypes import Player, Move
from go.utils import print_board, print_move, point_from_coords


def main():
    board_size = 9
    game = GameState.new_game(board_size)
    bot = RandomBot()

    while not game.is_over():
        print(chr(27) + "[2J")
        print_board(game.board)
        if game.next_player == Player.black:
            human_move = input("-- ")
            point = point_from_coords(human_move.strip())
            move = Move.play(point)
        else:
            move = bot.select_move(game)
        print_move(game.next_player, move)
        game = game.apply_move(move)


if __name__ == "__main__":
    main()
