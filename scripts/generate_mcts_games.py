#!/usr/bin/env python
import argparse
from typing import Tuple
import os
import sys

sys.path.insert(0, os.path.dirname(sys.argv[0]) + "/..")

import numpy as np
import numpy.typing as nptype

from go.encoders.encoder import get_encoder_by_name

# in the book they import goboard_fast, but I'm trying to avoid copying
# that until I've tried my own hand at going faster
from go.goboard import GameState
from go.agent import mcts
from go.utils import print_board, print_move


def generate_game(
    board_size: int, rounds: int, max_moves: int, temperature: float
) -> Tuple[nptype.NDArray[np.float64], nptype.NDArray[np.float64]]:
    boards, moves = [], []
    encoder = get_encoder_by_name("plane", board_size)
    game = GameState.new_game(board_size)
    # TODO: implement MCTSAgent
    bot = mcts.MCTSAgent(rounds, temperature)
    num_moves = 0
    while not game.is_over():
        print_board(game.board)
        move = bot.select_move(game)
        if move.is_play:
            boards.append(encoder.encode(game))
            move_one_hot = np.zeros(encoder.num_points())
            move_one_hot[encoder.encode_point(move.point)] = 1
            moves.append(move_one_hot)
        print_move(game.next_player, move)
        game = game.apply_move(move)
        num_moves += 1
        if num_moves > max_moves:
            break
    return np.array(boards), np.array(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", "-b", type=int, default=9)
    parser.add_argument("--rounds", "-r", type=int, default=1000)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument(
        "--max-moves", "-m", type=int, default=60, help="max moves per game"
    )
    parser.add_argument("--num-games", "-n", type=int, default=10)
    parser.add_argument("--board-out", help="name of the file to write boards to")
    parser.add_argument("--move-out", help="name of the file to write moves to")

    args = parser.parse_args()

    xs = []
    ys = []

    for i in range(args.num_games):
        print(f"Generating game {i+1}/{args.num_games}")
        x, y = generate_game(
            args.board_size, args.rounds, args.max_moves, args.temperature
        )
        xs.append(x)
        ys.append(y)

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    np.save(args.board_out, x)
    np.save(args.move_out, y)


if __name__ == "__main__":
    main()
