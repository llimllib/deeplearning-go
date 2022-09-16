#!/usr/bin/env python
from go.agent import RandomBot
from go.goboard_slow import GameState
from go.gotypes import Player
from go.utils import print_board, print_move
import time


def main():
    board_size = 9
    game = GameState.new_game(board_size)
    bots = {
        Player.black: RandomBot(),
        Player.white: RandomBot(),
    }
    while not game.is_over():
        # dont' play the bot moves super fast
        time.sleep(0.3)

        # clear screen
        print(chr(27) + "[2J")
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)


if __name__ == "__main__":
    main()
