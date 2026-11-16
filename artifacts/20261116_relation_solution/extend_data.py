"""Frozen external IDs and matched relation swaps; never select by model score."""
import sys,json,time,copy
from pathlib import Path
from hashlib import sha256
from urllib.request import Request,urlopen
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.relation_solution_data import intervene,targets,environment
from experiments.receipt_dataset import make_world
from kci.dynamic_tasks import canonical
REV='ac6634013dc74b1c9601d9e59248a59465438a43'

def fetch(i,name):
    assert 10<=i<=17 and name in ('scenario','utility')
    remote=f'meeting_allocation/scenario_{i:02d}/{name}.json'
    url=f'https://huggingface.co/datasets/ServiceNow/PiSAs/resolve/{REV}/{remote}'
    folder=R/'heldout_sources'/f'scenario_{i:02d}';folder.mkdir(parents=True,exist_ok=True)
    p=folder/(name+'.json');assert not p.exists();t=time.perf_counter()
    with urlopen(Request(url,headers={'User-Agent':'KCI-relation-audit/1.0'}),timeout=7) as response:raw=response.read(500001)
    assert len(raw)<=500000;json.loads(raw);p.write_bytes(raw)
    (folder/(name+'_receipt.json')).write_text(json.dumps({'url':url,'revision':REV,'sha256':sha256(raw).hexdigest(),'bytes':len(raw),'seconds':time.perf_counter()-t},indent=2))

def qualify(i):
    folder=R/'heldout_sources'/f'scenario_{i:02d}';folder.mkdir(parents=True,exist_ok=True);adapter=folder/'adapter'
    record={'scenario':i,'native_CI_obligation':False};worlds=[];states=[]
    try:
        files={}
        for name in ('scenario','utility'):
            raw=(folder/(name+'.json')).read_bytes();relative=f'pisas_hf/meeting_allocation/scenario_01/{name}.json'
            target=adapter/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            files[relative]={'sha256':sha256(raw).hexdigest(),'bytes':len(raw),'original_source_path':f'meeting_allocation/scenario_{i:02d}/{name}.json'}
        (adapter/'manifest.json').write_text(json.dumps({'files':files},indent=2))
        utility=json.loads((folder/'utility.json').read_text());types={c['type'] for c in utility['hidden_private_constraints']}
        record['constraint_types']=sorted(types)
        if types-{'cannot_use_room','person_unavailable','meeting_ordering'}:raise ValueError('unsupported native constraints: '+str(types))
        for j in range(2):
            w=make_world(20000000+i*100+j,'meeting',depth=2,noise=2)
            w.update(kernel_variant='ordering' if 'meeting_ordering' in types else 'original',source_instance_group=f'pisas_meeting_{i:02d}',source_root_relative=adapter.relative_to(ROOT).as_posix(),base_group_id=w['world_id'])
            env=environment(w);assert env.expected==utility['oracle_schedule'];assert env.exact(env.initial())['safe_success']
            worlds.append(w);states.extend(targets(w))
        record.update(status='COMPUTATION_QUALIFIED',source_files=files)
    except Exception as e:record.update(status='INCOMPATIBLE_OR_MISSING',reason=repr(e));worlds=[];states=[]
    out=R/'dataset/external';out.mkdir(exist_ok=True)
    for name,rows in [('worlds',worlds),('states',states)]:
        (out/f'{name}_{i:04d}.jsonl').write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    (folder/'qualification.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))

def swaps():
    for split in ('test','depth4','depth6','depth10','external'):
        folder=R/'dataset'/split
        for p in sorted(folder.glob('worlds_*.jsonl')):
            out=R/'dataset'/(split+'_swap');out.mkdir(exist_ok=True);target=out/p.name
            if target.exists():raise FileExistsError(target)
            target.write_text(''.join(canonical(intervene(json.loads(line),'swap'))+'\n' for line in p.read_text().splitlines()),encoding='utf-8')

if __name__=='__main__':
    if sys.argv[1]=='fetch':fetch(int(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=='qualify':qualify(int(sys.argv[2]))
    else:swaps()
