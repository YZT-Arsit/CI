"""Audit completed query generation against independent labels and seal artifacts."""
import json,sys,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from audit_data import check
from kci.relation_language import prompt,seed_for
from kci.action_binding_metrics import pair_metrics

def run():
    folder=R/'query_pilot';plan=json.loads((folder/'PLAN.json').read_text());assert hashlib.sha256((R/'query_pilot.py').read_bytes()).hexdigest()==plan['source_sha256']
    data=json.loads((folder/'dataset.json').read_text(encoding='utf-8'));cases=[r for pair in data['train'] for r in pair]+[r for pairs in data['dev'].values() for pair in pairs for r in pair]
    assert not [e for row in cases for e in check(row)];pool=json.loads((OLD/'entity_palettes.json').read_text())['fresh'];results=json.loads((folder/'RESULTS.json').read_text());count=0
    for report in results['models']:
        out=folder/report['arm'];raw=list(map(json.loads,(out/'dev_raw.jsonl').read_text(encoding='utf-8').splitlines()));scored=json.loads((out/'dev_scored.json').read_text());assert len(raw)==len(scored)==96
        for a,b in zip(raw,scored):
            row=data['dev'][str(a['depth'])][a['group']][a['side']];candidate=row['candidates'][a['candidate']];text,_=prompt(row['observation'],pool,seed_for(f"query_dev_{a['depth']}_{a['group']}",7331),question='task',candidate=candidate['action']);text=text.replace('A yes; B no.','Reply ANSWER: followed by 1 if possible, otherwise 0.')
            assert text==a['prompt'];matches=re.findall(r'ANSWER:\s*([01])\b',a['raw']);selected=matches[0] if len(matches)==1 and a['seconds_batch']<9.5 else None
            assert b['selected']==selected and b['optimal_action']==str(candidate['safe_completion']);assert b['pair_id']==f"d{a['depth']}_g{a['group']}_c{a['candidate']}" and b['side']==a['side'];count+=1
        for depth in (1,2,3):assert pair_metrics([r for r in scored if r['depth']==depth])==report['depths'][str(depth)]
        assert all(hashlib.sha256((out/'adapter'/name).read_bytes()).hexdigest()==value for name,value in report['adapter_hashes'].items())
    result={'independent_label_states':len(cases),'raw_generations_checked':count,'errors':0,'pilot_only':True};(folder/'AUDIT.json').write_text(json.dumps(result,indent=2))
    files=[p for p in folder.rglob('*') if p.is_file() and p.name!='COMPLETION_MANIFEST.json']
    dependencies=[R/'query_pilot.py',R/'probe_frozen.py',OLD/'lm_common.py',OLD/'proof_targets.py',ROOT/'code/experiments/relation_solution_data.py',ROOT/'code/experiments/receipt_dataset.py',ROOT/'code/kci/relation_language.py',ROOT/'code/kci/action_binding_metrics.py']
    manifest={'scope':'post-run byte integrity, not retroactive pre-run sealing','baseline_commit':plan.get('baseline_commit','b476787ba9e0cebf19d6b81569eeaf36b0570fbc'),'files':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files+dependencies}}
    (folder/'COMPLETION_MANIFEST.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
