import sys,json,copy,random
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import make_world,SOURCE
from kci.native_result_tasks import NativeResultTask,result_edges
from kci.dynamic_tasks import canonical

def run(index):
    kind='query' if index==1 else 'meeting'
    source=SOURCE if index<2 else R/f'heldout_sources/scenario_{index:02d}/adapter'
    prototype=make_world(420000+index*100,kind);prototype['kernel_variant']='ordering' if index>=6 else 'original'
    env=NativeResultTask(prototype,source);s=env.initial()
    actual=next(m['content']['result'] for m in s.ledger if m['content'].get('type')=='computed_native_result')
    assert actual==env.expected
    edges=result_edges(kind,actual);selected=[]
    for e in edges:
        if not selected or e[1]!=selected[0][1]:selected.append(e)
        if len(selected)==2:break
    report={'source_index':index,'native_result_matches_oracle':True,'extra_receipt_requirement':False,'available_distinct_native_values':len(selected)}
    out=R/'native_output';out.mkdir(exist_ok=True)
    if len(selected)!=2:
        report['status']='INAPPLICABLE_TWO_NATIVE_ROWS_REQUIRED';(out/f'source_{index:02d}.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    worlds=[];swaps=[];audits=[]
    for i in range(8):
        w=make_world(420000+index*100+i,kind);w['kernel_variant']=prototype['kernel_variant'];w['engine_variant']='native_result'
        w['providers']=[{'ticket':e[0],'links':[e]} for e in selected];random.Random(i).shuffle(w['providers'])
        w['histories']=[[selected[i%2][1],w['histories'][0][1]]];w['source_root_relative']=str(source.relative_to(ROOT)).replace('\\','/')
        w['source_instance_group']=f'native_output_source_{index:02d}';worlds.append(w)
        swapped=copy.deepcopy(w);swapped['histories'][0][0]=selected[1-i%2][1];swaps.append(swapped)
        pair=[]
        for world in (w,swapped):
            task=NativeResultTask(world,source);start=task.initial();proof=task.preference(start,*task.actions(start))
            assert proof['immediate']==['VALID','VALID'] and sorted(proof['values'])==[(1,0,0),(1,1,0)]
            obs=task.observation(start)
            assert obs['contract']['providers']==world['providers']
            assert not obs['contract']['extra_receipt_requirement']
            for a in task.actions(start):
                after,_=task.transition(start,a);final,audit=task.transition(after,'PUBLISH_FULL')
                assert final.ledger[-1]['content']=={'type':'native_result','result':actual}
            pair.append(proof)
        assert pair[0]['chosen']!=pair[1]['chosen'];audits.append({'world_id':w['world_id'],'proofs':pair})
    for name,rows in [('native_base',worlds),('native_swapped',swaps)]:
        folder=R/'dataset'/name;folder.mkdir(exist_ok=True)
        (folder/f'worlds_{index:04d}.jsonl').write_text(''.join(canonical(x)+'\n' for x in rows))
    report.update(status='QUALIFIED',worlds=len(worlds),paired_audits=audits)
    (out/f'source_{index:02d}.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='paired_audits'}))

if __name__=='__main__':run(int(sys.argv[1]))
