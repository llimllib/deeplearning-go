#!/usr/bin/env python
# This script is used to generate zobrist.py. See Makefile
import os
import random
import sys

sys.path.insert(0, os.path.dirname(sys.argv[0]) + "/..")

from go.gotypes import Player, Point


MAX63 = 0x7FFFFFFFFFFFFFFF

table = {}
for row in range(1, 20):
    for col in range(1, 20):
        for state in (Player.black, Player.white):
            code = random.randint(1, MAX63)
            table[Point(row, col), state] = code

codes = ",\n".join(
    [f"    ({pt}, {state}): {hash_code}" for (pt, state), hash_code in table.items()]
)

print(
    f"""from typing import Dict, Tuple
from go.gotypes import Player, Point

__all__ = ["HASH_CODE", "EMPTY_BOARD"]

HASH_CODE: Dict[Tuple[Point, Player], int]  = {{
{codes}
}}

EMPTY_BOARD = 0"""
)
