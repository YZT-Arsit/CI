"""Restore this continuation's archive without replacing evidence."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [(ROOT / 'artifacts' / name).resolve() for name in
        ['20261117_optimization_probe', '20261117_optimization_repair', '20261117_grounded_norm_probe', '20261117_option_fallback']]


def restore_run(run):
    rows = json.loads((run / 'compression_manifest.json').read_text(encoding='utf-8'))['files']
    restored = 0
    for row in rows:
        source = (ROOT / row['compressed_path']).resolve()
        target = (ROOT / row['raw_path']).resolve()
        if not source.is_relative_to(run) or not target.is_relative_to(run):
            raise ValueError('archive outside declared run')
        compressed = source.read_bytes()
        if hashlib.sha256(compressed).hexdigest() != row['compressed_sha256']:
            raise ValueError('compressed hash mismatch')
        raw = gzip.decompress(compressed)
        if len(raw) != row['raw_bytes'] or hashlib.sha256(raw).hexdigest() != row['raw_sha256']:
            raise ValueError('original hash mismatch')
        if target.exists():
            if target.read_bytes() != raw:
                raise FileExistsError('different existing evidence: ' + str(target))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            restored += 1
    return len(rows), restored


def main():
    counts = [restore_run(run) for run in RUNS]
    print(json.dumps({'verified': sum(v for v, _ in counts), 'restored': sum(n for _, n in counts)}))


if __name__ == '__main__':
    main()
