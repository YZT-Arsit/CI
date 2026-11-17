"""Pure-data reconstruction of the existing one-hop training buckets."""
import json
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / '20261116_relation_solution/dataset/train'


def buckets():
    groups, result = {}, {}
    for path in sorted(DATA.glob('states_*.jsonl')):
        for row in map(json.loads, path.read_text(encoding='utf-8').splitlines()):
            obs = row['observation']
            if len(obs['contract']['providers'][0]['links']) != 1:
                continue
            groups.setdefault((row['base_group_id'], obs['role'], obs['committed_ticket']), []).append(row)
    for pair in groups.values():
        assert len(pair) == 2
        obs = pair[0]['observation']
        c = obs['contract']
        key = (obs['role'], c['permission'], c['binding_required'],
               any(x['safe_completion'] for row in pair for x in row['candidates']),
               any(not x['immediate_safe'] for row in pair for x in row['candidates']))
        result.setdefault(key, []).append(sorted(pair, key=lambda row: row['world_id']))
    return result


def strict_reversal(pair):
    if not all(len(row['optimal_indices']) == 1 for row in pair):
        return False
    return pair[0]['candidates'][pair[0]['optimal_indices'][0]]['action'] != pair[1]['candidates'][pair[1]['optimal_indices'][0]]['action']
