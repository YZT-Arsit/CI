"""Varied one-round graphs and explicit sets of valid copied witnesses."""
import copy
import itertools
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_step_replay'
sys.path.insert(0, str(PARENT))
from replay_data import NAMES, read, rows, sha, verify as verify_parent, replay_schedule


def witnesses(world):
    return set()


def truth(world):
    n = world['n']; vector = [int(i in world['frontier']) for i in range(n)]
    matrix = [[int([i, j] in world['edges']) for j in range(n)] for i in range(n)]
    return int(vector[world['target']] or sum(vector[i] * matrix[i][world['target']] for i in range(n)))
