"""Verify evidence bindings and inventory this pilot; never rerun or alter evidence."""
from datetime import datetime, timezone
from hashlib import sha256
import difflib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parent
WORK = RUN.parent.parent


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(name, value):
    (RUN / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


original = read(RUN / 'input_manifest.json')
changes = []
preserved = []
backups = []
patches = []
original_paths = set()
for item in original['files']:
    rel = Path(item['path'])
    original_paths.add(rel)
    current = WORK / rel
    current_hash = digest(current)
    protected = rel.parts[0] in {'paper', 'docs'} or rel.name == 'SPEC_v0.5.md'
    if protected:
        assert current_hash == item['sha256'], 'Protected research input changed: ' + str(rel)
        preserved.append(str(rel))
    if rel.parts[0] in {'code', 'docs'}:
        backup = RUN / 'baseline' / rel
        assert digest(backup) == item['sha256'], 'Baseline does not match input: ' + str(rel)
        backups.append(str(rel))
        if current_hash != item['sha256']:
            changes.append({'path': str(rel), 'status': 'modified', 'before': item['sha256'], 'after': current_hash})
            patches.extend(difflib.unified_diff(backup.read_text(encoding='utf-8').splitlines(True),
                          current.read_text(encoding='utf-8').splitlines(True),
                          fromfile='baseline/' + rel.as_posix(), tofile=rel.as_posix()))
for folder in ('code', 'docs'):
    for path in sorted((WORK / folder).rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts:
            continue
        rel = path.relative_to(WORK)
        if rel not in original_paths:
            changes.append({'path': str(rel), 'status': 'added', 'after': digest(path)})
            if path.suffix in {'.py', '.md', '.toml', '.json'}:
                patches.extend(difflib.unified_diff([], path.read_text(encoding='utf-8').splitlines(True),
                              fromfile='/dev/null', tofile=rel.as_posix()))
write('changed_files.json', changes)
(RUN / 'changes.patch').write_text(''.join(patches), encoding='utf-8')

core = {p.name: digest(p) for p in sorted((WORK / 'code/kci').glob('*.py'))}
verified_records = 0
for path in sorted((RUN / 'dataset/shards').glob('*.verification.v2.json')):
    result = read(path)
    shard = path.with_name(path.name.replace('.verification.v2.json', '.jsonl'))
    assert result['input_sha256'] == digest(shard)
    assert result['core_sha256'] == core
    assert result['replay_results'] == {'verified': result['records']}
    verified_records += result['records']
assert verified_records == 2016
proxy_checks = 0
for folder in ('preference_proxy', 'preference_proxy_v2', 'order_probe_v2', 'balance_probe'):
    result = read(RUN / (folder + '_final_verification.json'))
    assert result['all_pass'] and result['core_sha256'] == core
    assert result['dataset_sha256'] == digest(RUN / 'dataset/canonical_preferences.jsonl')
    for check in result['checks']:
        assert check['replay_pass'] and check['sha256'] == digest(RUN / folder / check['file'])
        proxy_checks += 1
assert proxy_checks == 66
llm = read(RUN / 'ollama_agent/replay_audit_final.json')
assert llm['all_replays_passed'] and {Path(k).name: v for k, v in llm['core_file_sha256'].items()} == core
assert llm['requests_replayed'] == 62 and llm['executed_actions_replayed'] == 60
for name, expected in llm['raw_model_artifact_sha256'].items():
    assert digest(RUN / 'ollama_agent' / name) == expected, name
assert read(RUN / 'ollama_cleanup.json')['after']['models'] == []
assert (RUN / 'job_logs/final_full_regression.stderr.txt').read_text().rstrip().endswith('OK')
summary = read(RUN / 'results_summary.json')
assert summary['tests'] == 62 and summary['factor_cases'] == 187 and summary['proxy_policies'] == 66
assert summary['readiness']['bulk_llm_training'] == 'NO_GO'
optimizer = read(RUN / 'optimizer/summary.json')
assert optimizer['completed'] == 7 and all(optimizer['checks'].values())
optimizer_manifest = read(RUN / 'optimizer/manifest.json')
for name, expected in optimizer_manifest['source_and_input_sha256'].items():
    path = Path(name)
    if not path.is_absolute():
        path = WORK / 'code' / path
    assert digest(path) == expected, name
assert digest(RUN / 'optimizer/source_snapshot.py') == digest(WORK / 'code/experiments/optimizer_probe.py')
tokenization = read(RUN / 'tokenization/summary.json')
assert tokenization['status'] == 'PASS' and tokenization['shared_prompt_verified_pairs'] == 1824
assert tokenization['input_sha256'] == digest(RUN / 'dataset/preferences.jsonl')
assert tokenization['input_sha256'] == digest(RUN / 'tokenization/input_snapshot.jsonl')
assert tokenization['source_sha256'] == digest(WORK / 'code/experiments/tokenization_probe.py')
assert tokenization['source_sha256'] == digest(RUN / 'tokenization/source_snapshot.py')
token_config = read(RUN / 'tokenization/config.json')
for name, expected in token_config['tokenizer_files'].items():
    assert digest(RUN / 'tokenization/official_tokenizer' / name) == expected, name
for name, metadata in token_config['wheels'].items():
    assert digest(RUN / 'tokenization/wheels' / name) == metadata['sha256'], name
causal = read(RUN / 'tokenization/causal_boundary.json')
assert causal['history_causes_ci_label_flip_in_declared_toy'] and causal['matched_original_dataset_prompt']
assert causal['source_input_sha256'] == digest(RUN / 'dataset/preferences.jsonl')
for name, expected in causal['source_files_sha256'].items():
    assert digest(WORK / name) == expected, name
now = datetime.now(timezone.utc)
start = datetime.fromisoformat(original['started_utc'])
validation = {
    'validated_utc': now.isoformat(), 'minutes_since_input_snapshot': (now-start).total_seconds()/60,
    'all_checks_passed': True, 'original_protected_files_unchanged': preserved,
    'original_code_and_docs_backups_verified': len(backups), 'core_sha256': core,
    'synthetic_records_bound_to_current_core': verified_records,
    'finite_policies_bound_to_current_core': proxy_checks,
    'model_requests_replayed': llm['requests_replayed'],
    'model_actions_replayed': llm['executed_actions_replayed'],
    'raw_model_artifacts_hash_verified': len(llm['raw_model_artifact_sha256']),
    'tokenized_unique_pairs': tokenization['shared_prompt_verified_pairs'],
    'finite_optimizer_diagnostics': optimizer['completed'],
    'research_objective_changed': False, 'bulk_training_readiness': 'NO_GO',
    'scope': 'artifact integrity and previously executed evidence; not a new scientific replication',
}
write('FINAL_VALIDATION.json', validation)
inventory = []
for base in (WORK / 'code', WORK / 'docs', RUN):
    for path in sorted(base.rglob('*')):
        if (not path.is_file() or '__pycache__' in path.parts or
                path.name == 'final_manifest.json' or 'dependencies' in path.parts or 'vendor' in path.parts):
            continue
        inventory.append({'path': path.relative_to(WORK).as_posix(),
                          'bytes': path.stat().st_size, 'sha256': digest(path)})
write('final_manifest.json', {'generated_utc': now.isoformat(), 'files': inventory,
                            'exclusions': ['final_manifest.json itself', '__pycache__', 'dependencies', 'vendor']})
print(json.dumps({k: validation[k] for k in ('all_checks_passed', 'synthetic_records_bound_to_current_core',
                    'finite_policies_bound_to_current_core', 'raw_model_artifacts_hash_verified')}, ensure_ascii=False))
