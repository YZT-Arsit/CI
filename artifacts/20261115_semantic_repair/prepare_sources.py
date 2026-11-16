"""Qualify all eight preselected source instances; retain incompatibilities."""
import json,sys
from pathlib import Path
from hashlib import sha256

R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import make_world
from kci.receipt_tasks import ReceiptTask
from kci.dynamic_tasks import canonical

reports=[];worlds=[]
for scenario in range(2,10):
    folder=R/'heldout_sources'/f'scenario_{scenario:02d}';adapter=folder/'adapter'
    record={'scenario':scenario,'selection':'fixed before download/model evaluation','native_CI_obligation':False}
    try:
        files={}
        for name in ('scenario','utility'):
            raw=(folder/(name+'.json')).read_bytes();relative=f'pisas_hf/meeting_allocation/scenario_01/{name}.json'
            target=adapter/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            files[relative]={'sha256':sha256(raw).hexdigest(),'bytes':len(raw),
                'original_source_path':f'meeting_allocation/scenario_{scenario:02d}/{name}.json'}
        (adapter/'manifest.json').write_text(json.dumps({'files':files},indent=2))
        utility=json.loads((folder/'utility.json').read_text())
        probe=make_world(120000+scenario*100,'meeting')
        env=ReceiptTask(probe,adapter)
        assert env.expected==utility['oracle_schedule']
        record.update(status='COMPUTATION_QUALIFIED',source_scenario_id=utility['scenario_id'],
            constraint_types=sorted({c['type'] for c in utility['hidden_private_constraints']}),
            task_oracle_matches_native=True,source_files=files)
        for index in range(8):
            w=make_world(120000+scenario*100+index,'meeting',depth=1+index%3,noise=index%4)
            w['source_instance_group']=f'pisas_meeting_scenario_{scenario:02d}'
            w['source_root_relative']=str(adapter.relative_to(ROOT)).replace('\\','/')
            worlds.append(w)
    except Exception as error:record.update(status='INCOMPATIBLE_OR_MISSING',reason=repr(error))
    reports.append(record)
out=R/'dataset/source_holdout';out.mkdir(exist_ok=True)
(out/'worlds_0000.jsonl').write_text(''.join(canonical(w)+'\n' for w in worlds),encoding='utf-8')
(R/'source_qualification.json').write_text(json.dumps({'preselected':8,'records':reports,
    'qualified':sum(r['status']=='COMPUTATION_QUALIFIED' for r in reports),'worlds':len(worlds),
    'source_obligation_validation':False,'all_results_reported':True},indent=2))
print(json.dumps({'preselected':8,'statuses':[{k:r[k] for k in ('scenario','status')} for r in reports],'worlds':len(worlds)}))
