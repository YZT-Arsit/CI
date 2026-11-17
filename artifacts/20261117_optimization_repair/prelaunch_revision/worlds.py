"""Freeze teacher-estimate ablations with unchanged sampler and new DEV worlds."""
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
sys.path[:0] = [str(ROOT / 'code'), str(R.parent / '20261116_relation_solution'), str(R.parent / '20261117_action_binding')]
sys.path.insert(0, str(R.parent / '20261117_sampler_repair'))
from sampling import buckets, strict_reversal, DATA
from experiments.receipt_dataset import make_world
from experiments.relation_solution_data import intervene, targets
from build_structural_cases import pair as internal_pair
from build_degree_controls_v2 import anchored_pair
from audit_data import check


def dev_worlds():
    result, paired = {}, []
    for family, count in [('base', 8), ('depth3', 4), ('internal3', 4), ('cycle4', 4)]:
        for view in (0, 1):
            key = family + '_v' + str(view)
            result[key], result[key + '_swap'] = [], []
            paired.append(key)
        for n in range(count):
            seed = 350000000 + {'base': 0, 'depth3': 1000, 'internal3': 2000, 'cycle4': 3000}[family] + n
            kind = ('meeting', 'query')[n % 2]
            if family == 'internal3':
                pair = internal_pair(seed, 3, kind)
            elif family == 'cycle4':
                pair = anchored_pair(seed, 4, kind)
            else:
                a = make_world(seed, kind, depth=3 if family == 'depth3' else 1, noise=2, mode='one', allow=False, required=True, delivered=True)
                a['base_group_id'] = a['world_id']
                pair = [a, intervene(a, 'swap')]
            for view in (0, 1):
                for side, world in enumerate(pair):
                    w = copy.deepcopy(world)
                    w['base_group_id'] += '_view' + str(view)
                    result[family + '_v' + str(view) + ('_swap' if side else '')].append(w)
    for condition in ['optional', 'authorized', 'draft', 'both']:
        for view in (0, 1):
            result[condition + '_v' + str(view)] = []
        for n in range(4):
            seed = 350010000 + {'optional': 0, 'authorized': 1000, 'draft': 2000, 'both': 3000}[condition] + n
            w = make_world(seed, ('meeting', 'query')[n % 2], depth=1, noise=2, mode='both' if condition == 'both' else 'one', allow=condition == 'authorized', required=condition != 'optional', delivered=condition != 'draft')
            for view in (0, 1):
                v = copy.deepcopy(w)
                v['base_group_id'] = v['world_id'] + '_view' + str(view)
                result[condition + '_v' + str(view)].append(v)
    return result, paired
