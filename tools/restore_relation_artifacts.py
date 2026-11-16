"""Restore this relation-learning run's JSONL with byte-for-byte verification."""
import gzip,json
from hashlib import sha256
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/20261116_relation_solution'

def main():
    records=json.loads((RUN/'compression_manifest.json').read_text())['files'];restored=0
    for row in records:
        source=(ROOT/row['compressed_path']).resolve();target=(ROOT/row['raw_path']).resolve()
        if not source.is_relative_to(RUN.resolve()) or not target.is_relative_to(RUN.resolve()):raise ValueError('archive outside declared run')
        compressed=source.read_bytes()
        if sha256(compressed).hexdigest()!=row['compressed_sha256']:raise ValueError('compressed hash mismatch')
        raw=gzip.decompress(compressed)
        if len(raw)!=row['raw_bytes'] or sha256(raw).hexdigest()!=row['raw_sha256']:raise ValueError('original hash mismatch')
        if target.exists():
            if target.read_bytes()!=raw:raise FileExistsError('different existing evidence: '+str(target))
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);restored+=1
    print(json.dumps({'verified':len(records),'restored':restored}))

if __name__=='__main__':main()
