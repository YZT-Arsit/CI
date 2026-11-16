"""Serialize dynamic finite actions through the existing strict training adapter."""
import argparse,json
from pathlib import Path
from collections import Counter
from kci.contracts import Observation,Fact,Message,Action
from kci.training_data import PreferenceCorpusIndex,audit_record
from kci.dynamic_tasks import canonical


def export(source,output):
    source=Path(source);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    groups=[json.loads(s) for s in (source/'history_pairs.jsonl').read_text(encoding='utf-8').splitlines()]
    primary_cases={c for g in groups for c in g['cases']}
    rows=[];primary=[];index=PreferenceCorpusIndex()
    for line in (source/'preferences.jsonl').read_text(encoding='utf-8').splitlines():
        raw=json.loads(line);o=json.loads(raw['prompt'])
        task=canonical({k:o[k] for k in ('task','phase','protocol','actions','received_shards')})
        facts=(Fact('routing','routing','protocol',canonical(o['authorized_routing_metadata']),canonical(o['authorized_routing_metadata'])),)
        purpose='source_adapted_'+raw['workflow']
        receipts=tuple(Message(str(m['event_id']),m['sender'],m['recipient'],purpose,(),canonical(m['content'])) for m in o['authorized_receipts'])
        obs=Observation(raw['workflow']+'_source_instance_01','dynamic_norm_v1','dynamic_task_v1',o['role'],
                        task,purpose,len(raw['prefix']),o['remaining_steps'],facts,receipts,
                        (o['norm'],o['derived_relation_permission'],'Each executable action costs one unit.'),
                        False,o['phase']=='notify')
        target=o['phase']=='choose' and raw['case_id'] in primary_cases
        row={'prompt':obs.as_prompt(),'chosen':Action(o['role'],raw['chosen']).as_json(),
             'rejected':Action(o['role'],raw['rejected']).as_json(),
             'metadata':{'certified':True,'scenario_group_id':raw['workflow']+'_source_instance_01',
              'base_case_group_id':raw['workflow']+'_source_instance_01','workflow_group_id':raw['workflow'],
              'source_world_group_id':'explicit_source_adaptation_v1','rule_family_id':'three_relation_join',
              'split':'train','split_status':'no_independent_instance_holdout','source':raw['level'],
              'evidence_class':'exact_finite_source_adapted_synthetic_norm','norm_source':raw['norm_source'],
              'source_hashes':raw['source_hashes'],'human_verified':False,'natural_language_semantics_verified':False,
              'record_id':raw['preference_id'],'case_id':raw['case_id'],'state_id':raw['state_id'],
              'prefix':raw['prefix'],'proof':raw['proof'],'history_flip_primary':target,
              'input_contract':'Observation + canonical finite tool Action; no free-text send'}}
        audit_record(row);index.add(row);rows.append(row)
        if target:primary.append(row)
    for name,data in [('all_preferences',rows),('history_primary',primary)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data),encoding='utf-8')
    summary={'all_rows':len(rows),'primary_history_rows':len(primary),'workflows':dict(Counter(r['metadata']['workflow_group_id'] for r in primary)),
             'input_schema_pass':True,'independent_task_holdout':False,'primary_flip_groups':len(groups),
             'free_text_semantics':False,'human_verified':False}
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();export(a.source,a.output)
