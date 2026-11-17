"""Pinned tokenizer only: capacity and paired initial numerical-shape controls."""
import collections
import json
import sys
from step_data import R, sha, verify, cases
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env


def main():
    verify(); tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
    observations = cases()[0]; groups = collections.defaultdict(list); maxima = {}
    for mode in ['zero_shot', 'four_examples']:
        obs = [o for o in observations if o['mode'] == mode]; lengths = []
        for o in obs:
            lengths.append(len(tokenizer.apply_chat_template([{'role': 'user', 'content': o['prompt']}], tokenize=True, add_generation_prompt=True)))
        maxima[mode] = max(lengths)
        for i, o in enumerate(obs):
            if o['kind'] == 'switch':
                groups[mode, o['group'], o['vocab'], o['view']].append((lengths[i], max(lengths[(i // 4) * 4:(i // 4) * 4 + 4])))
    assert len(groups) == 256 and all(len(v) == 2 for v in groups.values())
    result = {'max_prompt_tokens': maxima, 'capacity': 1024, 'paired_groups': len(groups),
              'equal_own_and_padded_lengths': sum(len(set(v)) == 1 for v in groups.values()),
              'model_loaded': False, 'model_calls': 0, 'script_sha256': sha(R / 'token_preflight.py'), 'plan_sha256': sha(R / 'PLAN.json')}
    assert not (R / 'TOKEN_PREFLIGHT.json').exists()
    (R / 'TOKEN_PREFLIGHT.json').write_text(json.dumps(result, indent=2)); print(json.dumps(result), flush=True)
    assert max(maxima.values()) <= 1024 and result['equal_own_and_padded_lengths'] == 256


if __name__ == '__main__': main()
