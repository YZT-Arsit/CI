"""Offline audit/export of three pinned static source anchors; never trains or calls models."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

from kci.source_ingest import load_pisas_anchor,load_multiagentprivacy_anchor,SourceIngestError
from kci.training_data import audit_record,TrainingDataError


def run(root:Path):
    receipts=[json.loads(line) for line in (root/'download_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(r['status']=='downloaded' and r['worker_seconds']<9.5 for r in receipts)
    by_path={r['path']:r for r in receipts}
    assert len(by_path)==len(receipts)
    for path,receipt in by_path.items():assert sha256((root/path).read_bytes()).hexdigest()==receipt['sha256']
    hf=json.loads((root/'pisas_hf/repository.json').read_text(encoding='utf-8'))
    anchors=[];source_hashes={}
    for family in ('meeting_allocation','JIRA_allocation'):
        prefix=f'pisas_hf/{family}/scenario_01/'
        hashes={name:by_path[prefix+name]['sha256'] for name in
                ('scenario.json','utility.json','appropriateness.json','visibility.json')}
        source_hashes[family]=hashes
        anchors.append(load_pisas_anchor(root/prefix,revision=hf['sha'],workflow_group_id='pisas/'+family,
            expected_sha256=hashes,license_status='Apache-2.0 declared in pinned HF dataset card'))
    map_file='multiagentprivacy/data/experiment_10_def_adv_both.json'
    anchors.append(load_multiagentprivacy_anchor(root/map_file,
        revision='6e7af5d8a8b8a803d8d51cefc01831d2e0724b41',expected_sha256=by_path[map_file]['sha256'],
        license_status='No license found in reviewed pinned repository root or README; reuse unresolved'))
    checks=[]
    for anchor in anchors:
        assert anchor['delivered_events']==[] and anchor['derived_ci_labels']==[]
        assert not anchor['dynamic_train_ready'] and not anchor['human_verified'] and not anchor['model_input_ready']
        try:audit_record(anchor)
        except TrainingDataError:pass
        else:raise AssertionError('static anchor accepted as preference')
    checks+=['all three anchors have empty delivery/derived labels and false readiness',
             'all three anchors rejected by training sequence contract']
    directory=root/'pisas_hf/meeting_allocation/scenario_01'
    options=dict(revision=hf['sha'],workflow_group_id='pisas/meeting_allocation',
                 expected_sha256=source_hashes['meeting_allocation'],license_status='source declared Apache-2.0')
    bad=deepcopy(options);bad['expected_sha256']['scenario.json']='0'*64
    try:load_pisas_anchor(directory,**bad)
    except SourceIngestError:checks.append('modified source hash rejected')
    else:raise AssertionError('source hash bypass')
    for mutation in ('scenario_id','missing_atomic_label','visibility_overlap'):
        with tempfile.TemporaryDirectory(dir=root) as folder:
            target=Path(folder);hashes={}
            for path in directory.glob('*.json'):
                data=json.loads(path.read_text(encoding='utf-8'))
                if mutation=='scenario_id' and path.name=='utility.json':data['scenario_id']='wrong'
                if mutation=='missing_atomic_label' and path.name=='appropriateness.json':data['attributes'].pop('a1')
                if mutation=='visibility_overlap' and path.name=='visibility.json':data['attributes']['a1']['hidden_from'].append(data['attributes']['a1']['visible_to'][0])
                raw=json.dumps(data,ensure_ascii=False).encode('utf-8');(target/path.name).write_bytes(raw)
                hashes[path.name]=sha256(raw).hexdigest()
            altered={**options,'expected_sha256':hashes}
            try:load_pisas_anchor(target,**altered)
            except SourceIngestError:checks.append(mutation+' rejected')
            else:raise AssertionError('schema bypass: '+mutation)
    assert [len(a['initial_materials']['initial_artifacts']) for a in anchors[:2]]==[8,8]
    checks.append('all 16 timeline artifacts retained only as initial materials')
    (root/'anchors.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in anchors),encoding='utf-8')
    summary={'status':'PASS','anchor_count':3,'pisas_workflows':2,'delivered_events':0,
        'dynamic_preferences_produced':0,'human_verified':False,'checks':checks,
        'pisas':{a['workflow_group_id']:{**a['source_structural_audit'],
            'atomic_label_counts':dict(Counter(x['source_appropriateness'] for x in a['atomic_labels']))} for a in anchors[:2]},
        'multiagentprivacy':anchors[2]['source_structural_audit'],
        'download_bytes_by_source':dict(Counter({source:sum(r['bytes'] for r in receipts if r['path'].startswith(source+'/'))
            for source in ('pisas_hf','pisas_github','multiagentprivacy')})),
        'download_max_worker_seconds':max(r['worker_seconds'] for r in receipts),
        'source_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__),Path(__file__).parents[1]/'kci/source_ingest.py')},
        'output_sha256':sha256((root/'anchors.jsonl').read_bytes()).hexdigest(),
        'scope':'Static external source/schema/license audit, no dynamic labels, model requests, upload, or training.'}
    (root/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':run(Path(sys.argv[1]).resolve())
