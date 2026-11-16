"""Independent replay of native output transfer, without invoking a learned policy."""
import sys,json,copy
from pathlib import Path
from dataclasses import asdict
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from kci.native_result_tasks import NativeResultTask
from experiments.receipt_dataset import SOURCE
from kci.dynamic_tasks import canonical

def run(kind):
    folder=R/(kind+'_native_transfer');assert (folder/'EVALUATION_SEALED.json').exists();templates={};n=0;errors=[]
    for model in (p for p in folder.iterdir() if p.is_dir()):
        for file in model.glob('*.jsonl'):
            ws={w['world_id']:w for p in (ROOT/'artifacts/20261115_semantic_repair/dataset'/file.stem).glob('worlds_*.jsonl') for w in map(json.loads,p.read_text(encoding='utf-8').splitlines())}
            for row in map(json.loads,file.read_text(encoding='utf-8').splitlines()):
                w=ws[row['world_id']];source=ROOT/w['source_root_relative'] if 'source_root_relative' in w else SOURCE;key=(w['kind'],str(source))
                if key not in templates:templates[key]=NativeResultTask(w,source)
                e=copy.deepcopy(templates[key]);e.world=copy.deepcopy(w);e.binding['world']=copy.deepcopy(w);s=e.initial()
                for trace in row['trace']:
                    if trace['selected'] is None:break
                    s,audit=e.transition(s,trace['selected'])
                    if canonical(audit)!=canonical(trace['audit']):errors.append([model.name,w['world_id'],'transition'])
                if asdict(s)!=row['final']:errors.append([model.name,w['world_id'],'final'])
                for m in s.ledger:
                    if m['recipient']=='R' and m['content'].get('type')=='native_result':
                        if set(m['content'])!={'type','result'}:errors.append([model.name,w['world_id'],'extra output fields'])
                n+=1
    (R/(kind+'_native_replay.json')).write_text(json.dumps({'episodes':n,'errors':errors,'extra_native_fields':False},indent=2));print(n,len(errors));assert not errors

if __name__=='__main__':run(sys.argv[1])
