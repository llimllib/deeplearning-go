book github repo: https://github.com/maxpumperla/deep_learning_and_the_game_of_go/

currently something is busted, because bot_v_bot seems to go on forever

Errata:

- `legal_moves` and `winner` are in GameState and called but never given in the book
  - `compute_game_result` is called from `winner` and likewise never referenced
  - it's frustrating to try and follow along coding from the book when you
    end up just having to go to the github repo and getting it
- arbitrary class syntax
  - sometimes Class, Class(), or Class(Object)
  - minor irritation
- minor python nits
  - why not f-strings?
  - here they did float(x) / float(y) when only one of those is necessary
    - I get that this can be for clarity, just reads awk to me
    - return float(self.win_counts[player]) / self.num_rollouts
- MCTSAgent refers to values from its constructor that are not discussed in the book
  - num_rounds and temperature
  - it never gives a constructor for the agent at all
  - frustrating!
  - I wish they would say in the book that you need to be following along with the github... but the github also is encumbered by all the annotations for the book!
- book says `winning_pct` when `winning_frac` is the name of the function given earlier
  - [this line](https://github.com/maxpumperla/deep_learning_and_the_game_of_go/blob/6148f57eb98e4c75b102d096401efe780e911442/code/dlgo/mcts/mcts.py#L152) in the book is wrong
  - this error is present on both page 79 and page 81
- another error right below that, says `best_score = uct_score` but `uct_score` is a function defined just a few lines earlier
  - correct identifier is `score`
  - in the code (see link above) they've wisely inlined it to speed it up
- "The implementation of simulate_random_game is identical to the bot_v_bot example considered in chapter 3"
  - the implementation they're referring to is the `main` function, I found this surprising a bit, and had to discover what they meant by reading the github code.
- `np.doc` instead of `np.dot` on pg 92
  - https://github.com/maxpumperla/deep_learning_and_the_game_of_go/pull/105

TODO:

- benchmark goboard.py
  - compare against fast_goboard after you make your own attempt
  - test pypy
- benchmark randombot.py
  - see how far I can improve it
  - compare against code/dlgo/agent/naive_fast.py in the github repo
- Can we do better self-declarations than using strings?
  - example: MCTSNode has to type `parent` as `Optional['MCTSNode']` or else
    we get an error that it's undefined
