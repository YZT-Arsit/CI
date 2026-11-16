"""Lossless local JSONL archive; run only after all experiment writers finish."""
import gzip,json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]

def run():
    for name in ('language_evaluation','language_external_complete','balanced_language_evaluation','language_forced_states','balanced_language_forced_states','language_native_transfer'):
        if not (R/name/'EVALUATION_SEALED.json').exists():raise ValueError('unfinished experiment: '+name)
    if not (R/'time_budget_probe/RESULTS.json').exists():raise ValueError('unfinished timing probe')
    rows=[]
    for p in sorted(R.rglob('*.jsonl')):
        raw=p.read_bytes();blob=gzip.compress(raw,compresslevel=9,mtime=0);target=p.with_suffix(p.suffix+'.gz')
        if target.exists() and target.read_bytes()!=blob:raise FileExistsError('archive already exists with different bytes')
        target.write_bytes(blob);assert gzip.decompress(target.read_bytes())==raw
        rows.append({'raw_path':p.relative_to(ROOT).as_posix(),'compressed_path':target.relative_to(ROOT).as_posix(),'raw_bytes':len(raw),'compressed_bytes':len(blob),'raw_sha256':hashlib.sha256(raw).hexdigest(),'compressed_sha256':hashlib.sha256(blob).hexdigest()})
    (R/'compression_manifest.json').write_text(json.dumps({'files':rows,'raw_bytes':sum(r['raw_bytes'] for r in rows),'compressed_bytes':sum(r['compressed_bytes'] for r in rows)},indent=2))
    print(json.dumps({'files':len(rows),'raw_bytes':sum(r['raw_bytes'] for r in rows),'compressed_bytes':sum(r['compressed_bytes'] for r in rows)}))

if __name__=='__main__':run()
