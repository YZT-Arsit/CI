"""Restore the two bounded continuation archives without replacing evidence."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ['20261117_binding_factors', '20261117_native_roles']


def main():
    restored = verified = 0
    for name in RUNS:
        run = (ROOT / 'artifacts' / name).resolve()
        manifest = json.loads((run / 'compression_manifest.json').read_text(encoding='utf-8'))
        for row in manifest['files']:
            source = (ROOT / row['compressed_path']).resolve()
            target = (ROOT / row['raw_path']).resolve()
            if not source.is_relative_to(run) or not target.is_relative_to(run):
                raise ValueError('archive path outside declared run')
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
            verified += 1
    print(json.dumps({'verified': verified, 'restored': restored}))


if __name__ == '__main__':
    main()
