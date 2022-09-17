import random
from typing import List, Optional

from go.gotypes import Player
from go.goboard import GameState


class MCTSNode:
    def __init__(
        self, game_state: GameState, parent: Optional["MCTSNode"] = None, move=None
    ):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.win_counts = {
            Player.black: 0,
            Player.white: 0,
        }
        self.num_rollouts = 0
        self.children: List["MCTSNode"] = []
        self.unvisited_moves = game_state.legal_moves()

    def add_random_child(self) -> "MCTSNode":
        index = random.randint(0, len(self.unvisited_moves) - 1)
        new_move = self.unvisited_moves.pop(index)
        new_game_state = self.game_state.apply_move(new_move)
        new_node = MCTSNode(new_game_state, self, new_move)
        self.children.append(new_node)
        return new_node

    def record_win(self, winner: Player):
        self.win_counts[winner] += 1
        self.num_rollouts += 1

    def can_add_child(self) -> bool:
        return len(self.unvisited_moves) > 0

    def is_terminal(self) -> bool:
        return self.game_state.is_over()

    def winning_frac(self, player: Player) -> float:
        return float(self.win_counts[player]) / self.num_rollouts
