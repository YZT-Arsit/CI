"""Lossless reproducible compression; retain raw local evidence unchanged."""
import gzip,json
from pathlib import Path
from hashlib import sha256
R=Path(__file__).resolve().parent;ROOT=R.parents[1];records=[]
for p in sorted(R.rglob('*.jsonl')):
    raw=p.read_bytes();compressed=gzip.compress(raw,compresslevel=6,mtime=0);out=Path(str(p)+'.gz')
    if out.exists() and out.read_bytes()!=compressed:raise FileExistsError('different compression archive')
    out.write_bytes(compressed);assert gzip.decompress(out.read_bytes())==raw
    records.append({'raw_path':p.relative_to(ROOT).as_posix(),'compressed_path':out.relative_to(ROOT).as_posix(),
        'raw_bytes':len(raw),'compressed_bytes':len(compressed),'raw_sha256':sha256(raw).hexdigest(),'compressed_sha256':sha256(compressed).hexdigest()})
(R/'compression_manifest.json').write_text(json.dumps({'format':'gzip mtime0, original bytes verified','raw_files_retained_locally':True,'files':records},indent=2))
print(json.dumps({'files':len(records),'raw_bytes':sum(r['raw_bytes'] for r in records),'compressed_bytes':sum(r['compressed_bytes'] for r in records)}))
