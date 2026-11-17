"""Matched local batches plus balanced original-TRAIN Worker rehearsal."""
import collections
import copy
import importlib.util
import itertools
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent; ROOT = R.parents[1]; PARENT = R.parent / '20261117_step_learning'
sys.path.insert(0, str(PARENT))
from learning_data import NAMES, read, rows, sha, verify as verify_parent, train_worlds, training_schedule, confirmation_worlds, prompt, label, pair
sys.path.insert(0, str(ROOT / 'code'))
RENDER = R.parent / '20261117_binding_factors/render.py'
spec = importlib.util.spec_from_file_location('replay_actor_renderer', RENDER)
renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)


def replay_schedule(seed):
    return []


def worker_pairs():
    batches = rows(NAMES / 'batches.jsonl')
    result = [b['pairs'][str(b['mixed_depth'])] for b in batches if b['pairs'][str(b['mixed_depth'])][0]['observation']['role'] == 'Worker']
    assert len(result) == 6
    return result
