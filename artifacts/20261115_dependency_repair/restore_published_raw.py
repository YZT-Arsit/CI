"""Restore archived JSON byte-for-byte; never overwrite a different local file."""
from pathlib import Path
from hashlib import sha256
import gzip,json

R=Path(__file__).resolve().parent
manifest=json.loads((R/'published_raw/MANIFEST.json').read_text(encoding='utf-8'))
for record in manifest['files']:
    target=(R/record['source']).resolve()
    if not target.is_relative_to(R):raise ValueError('archive destination outside run directory')
    packed=(R/record['compressed']).read_bytes()
    if sha256(packed).hexdigest()!=record['compressed_sha256']:raise ValueError('compressed hash mismatch')
    data=gzip.decompress(packed)
    if sha256(data).hexdigest()!=record['raw_sha256']:raise ValueError('raw hash mismatch')
    if target.exists():
        if target.read_bytes()!=data:raise FileExistsError('different existing local evidence: '+str(target))
    else:
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
print(json.dumps({'verified_or_restored_files':len(manifest['files'])}))
