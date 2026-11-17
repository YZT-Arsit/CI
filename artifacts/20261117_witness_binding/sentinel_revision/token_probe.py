"""Check copied-name/sentinel token boundaries before fixing witness loss."""
import json
import sys
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parent / '20261116_relation_solution'))
from lm_common import AutoTokenizer, env
tokenizer = AutoTokenizer.from_pretrained(env['native_lm_path'], local_files_only=True)
pools = json.loads((R.parent / '20261117_rename_coverage/palettes.json').read_text())
words = set(pools['train'] + pools['new']) | {'NONE'}
failures = []
for word in sorted(words):
    atom = tokenizer.encode(' ' + word, add_special_tokens=False)
    encoded = tokenizer.encode('PROOF: ' + word + tokenizer.eos_token, add_special_tokens=False)
    if len(atom) != 1 or encoded[-2:] != [atom[0], tokenizer.eos_token_id]: failures.append(word)
assert 'NONE' not in pools['train'] + pools['new']
result = {'words_checked': len(words), 'failures': failures, 'model_loaded': False, 'model_calls': 0}
(R / 'TOKEN_PROBE.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result)); assert not failures
