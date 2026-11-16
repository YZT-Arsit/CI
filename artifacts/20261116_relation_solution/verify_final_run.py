"""Final consistency check of checkpoint/data hashes and recorded episode counters."""
import json,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run():
    checkpoints=0;files=0;episodes=0;errors=[]
    for folder in ('language_models','balanced_language_models'):
        for name,value in json.loads((R/folder/'plan.json').read_text(encoding='utf-8'))['source_hashes'].items():assert digest(R.parents[1]/name)==value
    for folder in ('structured_models','language_models','balanced_language_models'):
        for row in json.loads((R/folder/'CHECKPOINTS_SEALED.json').read_text(encoding='utf-8'))['models']:
            if 'adapter_hashes' in row:
                for name,value in row['adapter_hashes'].items():assert digest(R/folder/row['model']/'adapter'/name)==value
            else:assert digest(R/folder/row['model']/'model.pt')==row['checkpoint_sha256']
            checkpoints+=1
    for name,value in json.loads((R/'DATA_AUDIT.json').read_text(encoding='utf-8'))['files'].items():assert digest(R/name)==value;files+=1
    for folder in sorted(R.iterdir()):
        seal=folder/'EVALUATION_SEALED.json'
        if not folder.is_dir() or not seal.exists() or folder.name.endswith('_forced_states'):continue
        for summary in json.loads(seal.read_text(encoding='utf-8'))['summaries']:
            name=summary['condition'];path=folder/summary['model']/(name+'.jsonl')
            rows=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));assert len(rows)==summary['episodes']
            for key in ('ci','safe_success','refused','invalid'):
                if key in summary:assert sum(bool(r.get(key,False)) for r in rows)==summary[key],(folder.name,summary['model'],name,key)
            for row in rows:
                s=row['final'];invalid=row.get('invalid',False)
                if row['ci']!= (not s['violation']) or row['safe_success']!=(s['success'] and not s['violation'] and not invalid):errors.append([folder.name,summary['model'],name,row['world_id']])
            episodes+=len(rows)
    result={'sealed_primary_checkpoints':checkpoints,'dataset_files_unchanged':files,'episode_counters_checked':episodes,'errors':errors,'scope':'consistency check; independent transition and proof checks recorded separately'}
    (R/'FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert not errors

if __name__=='__main__':run()
