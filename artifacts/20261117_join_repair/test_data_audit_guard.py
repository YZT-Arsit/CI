import json
import runpy
import sys
from pathlib import Path
folder = Path(__file__).resolve().parent
sys.path.insert(0, str(folder))
import join_data
original = join_data.rows
def corrupted(path):
    data = original(path)
    if path.name == 'schedule_17.jsonl':
        data[0]['labels'][0] = 1 - data[0]['labels'][0]
    return data
before = (folder / 'DATA_AUDIT.json').read_bytes()
join_data.rows = corrupted
try:
    runpy.run_path(str(folder / 'data_audit.py'))
except AssertionError as error:
    assert error.args and error.args[0][0] == 17
    assert (folder / 'DATA_AUDIT.json').read_bytes() == before
    (folder / 'DATA_AUDIT_GUARD.json').write_text(json.dumps({'in_memory_label_mutation_rejected': True, 'real_files_unchanged': True, 'model_calls': 0, 'failure': str(error)}), encoding='utf-8')
    print('PASS: corrupted label rejected before receipt write; real files unchanged.')
else:
    raise AssertionError('corrupted label accepted')
