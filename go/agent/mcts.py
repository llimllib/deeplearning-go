import math

from go.agent.base import Agent
from go.agent.naive import RandomBot
from go.goboard import GameState
from go.gotypes import Player, Move
from go.mcts import MCTSNode


def uct_score(parent_rollouts, child_rollouts, win_pct, temperature):
    exploration = math.sqrt(math.log(parent_rollouts) / child_rollouts)
    return win_pct + temperature * exploration


class MCTSAgent(Agent):
    def __init__(self, num_rounds: int, temperature: float):
        Agent.__init__(self)
        self.num_rounds = num_rounds
        self.temperature = temperature

    def select_move(self, game_state: GameState) -> Move:
        root = MCTSNode(game_state)

        for _ in range(self.num_rounds):
            node = root
            while (not node.can_add_child()) and (not node.is_terminal()):
                node = self.select_child(node)

            if node.can_add_child():
                node = node.add_random_child()

            winner = self.simulate_random_game(node.game_state)

            while node is not None:
                node.record_win(winner)
                node = node.parent

            best_move = None
            best_pct = -1.0
            for child in root.children:
                child_pct = child.winning_frac(game_state.next_player)
                if child_pct > best_pct:
                    best_pct = child_pct
                    best_move = child.move

            assert best_move
            return best_move

        raise ValueError("unreachable code")

    def select_child(self, node: MCTSNode) -> MCTSNode:
        total_rollouts = sum(child.num_rollouts for child in node.children)

        best_score: float = -1.0
        best_child = None
        for child in node.children:
            score = uct_score(
                total_rollouts,
                child.num_rollouts,
                child.winning_frac(node.game_state.next_player),
                self.temperature,
            )
            if score > best_score:
                best_score = score
                best_child = child

        assert best_child
        return best_child

    @staticmethod
    def simulate_random_game(game) -> Player:
        bots = {
            Player.black: RandomBot(),
            Player.white: RandomBot(),
        }
        while not game.is_over():
            bot_move = bots[game.next_player].select_move(game)
            game = game.apply_move(bot_move)
        return game.winner()
