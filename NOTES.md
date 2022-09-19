book github repo: https://github.com/maxpumperla/deep_learning_and_the_game_of_go/

currently something is busted, because bot_v_bot seems to go on forever

Notes:

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
- random usage of the `six` module despite limiting to python 3
  - `input` is `input` in all versions of python 3, for example
- so many uses of "just"... for things that are quite complex!
- the book shows using pickle to load the mnist data, but it's stored in the repo as an `npz` file and the github code has somewhat been modified to match, but not entirely

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
- add tests
- figure out how to get pyright to see numpy
  - need to get "type stubs" in there somehow
- my numpy typings are a mess, make them better
  - numpy.typing.NDArray[float64] sort of types maybe?
  - related to the above TODO
- Benchmark chapter 4 nn
  - profile and improve it
  - interesting note: pypy doesn't suck up all my CPUs but regular python seems to do
    - could be interesting to dig into!
  - implementing it with numba: https://www.bragitoff.com/2021/12/efficient-implementation-of-softplus-activation-function-and-its-derivative-gradient-in-python/

```
from numba import njit
@njit(cache=True,fastmath=True)
def Softplus(x):
    # Reference: https://stackoverflow.com/questions/44230635/avoid-overflow-with-softplus-function-in-python
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    # np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)
```

---

benchmarking chap 4 NN:

- ran it with python -mcProfile scripts/train_mnist.py -scumtime
  - that failed to sort by cumtime, but ok still was able to browse the results
  - looks like sigmoid is where we spent our data. hmm.
  - first thing I found suggested using `expit` from scipy:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.expit.html
    - let's write a test to make sure we're gonna get the same thing!
- command that sorts properly: python -m cProfile -s cumtime scripts/train_mnist.py

```
3676469 function calls (3670707 primitive calls) in 42.357 seconds

   Ordered by: cumulative time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    264/1    0.000    0.000   42.358   42.358 {built-in method builtins.exec}
        1    0.000    0.000   42.358   42.358 train_mnist.py:1(<module>)
        1    0.000    0.000   42.232   42.232 train_mnist.py:10(main)
        1    0.002    0.002   41.962   41.962 layer.py:240(train)
     4024    0.004    0.000   41.927    0.010 layer.py:263(train_batch)
     4024    0.191    0.000   34.730    0.009 layer.py:277(forward_backward)
   120699    9.801    0.000   31.010    0.000 layer.py:158(backward)
   362100    0.131    0.000   23.461    0.000 <__array_function__ internals>:177(dot)
432115/432104   23.311    0.000   23.361    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
     4023    0.018    0.000    7.193    0.002 layer.py:267(update)
    12069    6.429    0.001    6.429    0.001 layer.py:177(update_params)
   120701    0.209    0.000    2.550    0.000 layer.py:152(forward)
```
