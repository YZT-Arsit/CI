"""Repair source coverage on all preselected IDs, preserving first qualification failures."""
from extend_data import *

def run(i):
    folder=R/'heldout_sources'/f'scenario_{i:02d}';adapter=folder/'adapter'
    record={'scenario':i,'native_CI_obligation':False,'supersedes_coverage_only':'qualification.json','model_outputs_used_for_repair':False};worlds=[];states=[]
    try:
        utility=json.loads((folder/'utility.json').read_text());record['constraint_types']=sorted({c['type'] for c in utility['hidden_private_constraints']})
        for j in range(2):
            w=make_world(20000000+i*100+j,'meeting',depth=2,noise=2)
            w.update(kernel_variant='extended',source_instance_group=f'pisas_meeting_{i:02d}',source_root_relative=adapter.relative_to(ROOT).as_posix(),base_group_id=w['world_id'])
            env=environment(w);assert env.expected==utility['oracle_schedule'];assert env.exact(env.initial())['safe_success']
            worlds.append(w);states.extend(targets(w))
        record.update(status='COMPUTATION_QUALIFIED',oracle_match=True)
    except Exception as e:record.update(status='INCOMPATIBLE_OR_MISSING',reason=repr(e));worlds=[];states=[]
    out=R/'dataset/external_complete';out.mkdir(exist_ok=True)
    for name,rows in [('worlds',worlds),('states',states)]:
        path=out/f'{name}_{i:04d}.jsonl';assert not path.exists();path.write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
    out=R/'dataset/external_complete_swap';out.mkdir(exist_ok=True)
    (out/f'worlds_{i:04d}.jsonl').write_text(''.join(canonical(intervene(w,'swap'))+'\n' for w in worlds),encoding='utf-8')
    (folder/'qualification_extended.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))

if __name__=='__main__':run(int(sys.argv[1]))
