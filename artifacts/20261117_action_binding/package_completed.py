"""Losslessly package completed evidence and record a post-run integrity manifest."""
import gzip,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]

def digest(data):return hashlib.sha256(data).hexdigest()

def main():
    final=json.loads((R/'FINAL_RESULTS.json').read_text());assert final['total_new_adapters']==12 and final['total_optimizer_updates']==5200
    assert json.loads((R/'alias_confirmation/RESULTS.json').read_text())['errors']==0
    records=[]
    for p in sorted(R.rglob('*.jsonl')):
        raw=p.read_bytes();compressed=gzip.compress(raw,mtime=0);assert gzip.decompress(compressed)==raw;target=p.with_suffix(p.suffix+'.gz');target.write_bytes(compressed)
        records.append({'raw_path':p.relative_to(ROOT).as_posix(),'raw_bytes':len(raw),'raw_sha256':digest(raw),'compressed_path':target.relative_to(ROOT).as_posix(),'compressed_bytes':len(compressed),'compressed_sha256':digest(compressed)})
    (R/'compression_manifest.json').write_text(json.dumps({'format':'gzip; mtime0; original files retained locally','files':records},indent=2))
    paths=[p for p in R.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.gz','.md','.pyc') and not (p.suffix=='.log' and p.stat().st_size==0) and p.name not in ('compression_manifest.json','COMPLETION_MANIFEST.json')]
    dependencies=list((ROOT/'code').rglob('*.py'))+list((R.parent/'20261116_relation_solution').glob('*.py'))+[ROOT/'tools/restore_action_binding_artifacts.py']
    manifest={'scope':'Post-run integrity only, not a retroactive pre-run seal. Human Markdown reports and compressed files are covered by the upload/compression manifests. Referenced historical data remains in prior archives.','files':[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())} for p in sorted(set(paths+dependencies))]}
    (R/'COMPLETION_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'jsonl_files':len(records),'raw_bytes':sum(r['raw_bytes'] for r in records),'compressed_bytes':sum(r['compressed_bytes'] for r in records),'integrity_files':len(manifest['files'])}))

if __name__=='__main__':main()
