"""Build a third disjoint entity palette using the pinned tokenizer, without loading a model."""
import hashlib
import json
import random
import sys
from pathlib import Path
R = Path(__file__).resolve().parent
BASE = R.parent / '20261116_relation_solution'
sys.path.insert(0, str(BASE))
from lm_common import AutoTokenizer, env


def main():
    assert not (R / 'palettes.json').exists()
    tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
    words = []
    for token in tokenizer.get_vocab():
        word = token.replace('\u0120', '')
        if word.isascii() and word.isalpha() and word.islower() and 4 <= len(word) <= 6 and len(tokenizer.encode(' ' + word, add_special_tokens=False)) == 1:
            words.append(word)
    words = sorted(set(words)); random.Random(216031).shuffle(words)
    original = json.loads((BASE / 'entity_palettes.json').read_text())
    assert words[:128] == original['train'] and words[128:256] == original['fresh']
    pools = {'train': words[:128], 'known_fresh': words[128:256], 'new': words[256:384]}
    assert all(len(v) == 128 for v in pools.values()) and len(set(sum(pools.values(), []))) == 384
    (R / 'palettes.json').write_text(json.dumps(pools, indent=2), encoding='utf-8')
    receipt = {'palette_sha256': hashlib.sha256((R / 'palettes.json').read_bytes()).hexdigest(),
               'source_manifest_sha256': hashlib.sha256((BASE / 'lm_source_manifest.json').read_bytes()).hexdigest(),
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'token_ids': {k: [tokenizer.encode(' ' + w, add_special_tokens=False)[0] for w in v] for k, v in pools.items()},
               'model_loaded': False, 'model_calls': 0,
               'scope': 'Disjoint entity palettes in this experimental lineage, not unseen tokens in base-model pretraining.'}
    (R / 'PALETTE_RECEIPT.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps({'entity_names_per_palette': 128, 'pairwise_disjoint': True, 'model_calls': 0}))


if __name__ == '__main__':
    main()
