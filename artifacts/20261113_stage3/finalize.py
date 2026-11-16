"""Seal stage 3 with bounded hash workers and explicit evidence checks."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PREVIOUS = ROOT / 'artifacts/20261113_iteration2'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def record(p):
    return {'path': p.relative_to(ROOT).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}


def shards(paths, label):
    groups = [[]]
    size = 0
    for p in sorted(set(paths)):
        n = p.stat().st_size
        if groups[-1] and (len(groups[-1]) >= 100 or size + n > 32_000_000):
            groups.append([])
            size = 0
        groups[-1].append(p.relative_to(ROOT).as_posix())
        size += n
    rows, receipts = [], []
    for i, group in enumerate(groups):
        request = RUN / 'sealing' / f'{label}_{i:03}.request.json'
        response = request.with_suffix('.response.json')
        write(request, group)
        start = time.perf_counter()
        subprocess.run([sys.executable, '-B', str(Path(__file__).resolve()), '--hash',
                        str(request), str(response)], cwd=ROOT, check=True, timeout=9.5)
        rows.extend(read(response))
        receipts.append({'shard': i, 'files': len(group), 'seconds': time.perf_counter()-start,
                         'hard_timeout_seconds': 9.5, 'status': 'completed'})
    return rows, receipts


def finalize():
    (RUN/'sealing').mkdir(exist_ok=True)
    baseline = PREVIOUS/'final_manifest.json'
    baseline_hash = digest(baseline)
    assert baseline_hash == '9eade5f78a313148f5467ae2a2bc0a31e056b03c1e37ade74eddf5b59fea7ffc'
    old = read(baseline)['files']
    expected = {x['path'].replace('\\', '/'): x['sha256'] for x in old}
    papers = read(PREVIOUS/'input_snapshot.json')['previous_paper_files']
    expected.update({x['path'].replace('\\', '/'): x['sha256'] for x in papers})
    protected, protection_receipts = shards([ROOT/p for p in expected], 'protected')
    assert all(x['sha256'] == expected[x['path']] for x in protected)
    checks = ['audit_task_meeting', 'audit_task_query', 'audit_cce_safe',
              'audit_cce_careless', 'audit_cce_mixed', 'finite_audit', 'qwen_audit', 'data_replay_audit']
    for name in checks:
        assert read(RUN/f'{name}.json')['all_passed'], name
    for requirement in read(RUN/'completion_audit.json'):
        for path in requirement['evidence']:
            assert (RUN/path).exists(), path
    manifest = read(RUN/'final_tests_source_manifest.json')['source_sha256']
    for path, sha in manifest.items():
        assert digest(ROOT/'code'/path) == sha, path
    receipt_names = ['final_tests', 'dynamic_cce_jobs_v3', 'finite_jobs_v2', 'model_audit_jobs',
                     'audit_jobs_v2', 'final_core_jobs', 'release_jobs']
    job_counts = {}
    for name in receipt_names:
        receipts = [json.loads(x) for x in (RUN/f'{name}_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
        assert receipts and all(x['status'] == 'completed' and x['source_hashes_unchanged']
                                and x['seconds'] <= 10 for x in receipts), name
        job_counts[name] = len(receipts)
    assert 'Ran 124 tests' in (RUN/'job_logs/final_124_tests.stderr.txt').read_text(encoding='utf-8')
    release = read(RUN/'qwen_v2/model_release.json')
    assert release['target_model_absent_after_release'] and not release['after']['ps']['models']
    assert read(RUN/'results_summary.json')['tests'] == 124
    sources = sorted(p for folder in [ROOT/'code', ROOT/'docs'] for p in folder.rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts)
    added = [p for p in sources if p.relative_to(ROOT).as_posix() not in expected]
    patch = []
    for p in added:
        name = p.relative_to(ROOT).as_posix()
        patch.extend(difflib.unified_diff([], p.read_text(encoding='utf-8').splitlines(keepends=True),
                                         fromfile='/dev/null', tofile=f'b/{name}'))
    (RUN/'changes.patch').write_text(''.join(patch), encoding='utf-8')
    write(RUN/'changed_files.json', {'existing_modified': [], 'added': [record(p) for p in added]})
    for p in sources:
        target = RUN/'final_source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    write(RUN/'FINAL_VALIDATION.json', {
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'all_passed': True,
        'previous_manifest_files_verified': len(old), 'paper_files_verified': len(papers),
        'all_distinct_protected_files_verified': len(protected), 'protected_mismatches': [],
        'previous_manifest_sha256': baseline_hash, 'regression_tests': 124,
        'final_test_source_hashes_match': True, 'evidence_checks': checks,
        'completion_evidence_paths_exist': True, 'validated_job_counts': job_counts,
        'ollama_released_at_recorded_cleanup': True, 'old_code_and_docs_modified': [],
        'new_code_and_docs_files': len(added), 'integrity_worker_receipts': protection_receipts,
        'scope': 'File integrity and same-researcher evidence consistency; not independent scientific validation',
        'large_method_experiments': 'NO_GO'})
    paths = sources + [p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                       and p.name != 'final_manifest.json'
                       and not (p.parent == RUN/'sealing' and p.name.startswith(('seal_', 'verify_')))]
    rows, receipts = shards(paths, 'seal')
    write(RUN/'final_manifest.json', {
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'path_base': str(ROOT),
        'files': rows, 'file_count': len(rows), 'total_bytes': sum(x['bytes'] for x in rows),
        'hash_worker_receipts': receipts, 'previous_manifest_sha256': baseline_hash,
        'exclusions': ['self', '__pycache__', 'sealing/seal_*', 'sealing/verify_*'],
        'scope': 'SHA256 change detection, not tamper-proof archival'})
    verified, verification_receipts = shards([ROOT/x['path'] for x in rows], 'verify')
    assert verified == rows
    print(json.dumps({'all_passed': True, 'protected_files': len(protected),
                      'new_source_files': len(added), 'manifest_files': len(rows),
                      'manifest_sha256': digest(RUN/'final_manifest.json'),
                      'max_integrity_worker_seconds': max(x['seconds'] for x in
                          protection_receipts + receipts + verification_receipts)}))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--hash':
        write(Path(sys.argv[3]), [record(ROOT/p) for p in read(Path(sys.argv[2]))])
    else:
        finalize()
