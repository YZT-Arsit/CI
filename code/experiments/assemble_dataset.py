"""Assemble audit data without confusing renamed copies with independent worlds."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

from experiments.common import write_result


def canonical_pattern(record):
    index=record['metadata']['world_config']['index']
    text='\n'.join(record[k] for k in ('prompt','chosen','rejected'))
    text=text.replace(f'K{1000+index}','TARGET_ID').replace(f'K{2000+index}','OTHER_ID')
    text=text.replace(f'Person{index:03d}','PERSON')
    return re.sub(r'\bB[123]\b','ELIGIBLE_BAND',text)


def assemble(folder):
    folder=Path(folder);raw=[];verified=0
    for shard in sorted((folder/'shards').glob('*.jsonl')):
        check=json.loads(shard.with_suffix('.verification.v2.json').read_text())
        if set(check['replay_results'])!={'verified'}:raise ValueError('Unverified shard')
        if check.get('input_sha256')!=sha256(shard.read_bytes()).hexdigest():raise ValueError('Verification does not bind current shard')
        core_root=Path(__file__).resolve().parents[1]/'kci'
        if check.get('core_sha256')!={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(core_root.glob('*.py'))}:
            raise ValueError('Verification core hashes are stale')
        verified+=check['replay_results']['verified']
        for line_no,line in enumerate(shard.read_text(encoding='utf-8').splitlines(),1):
            row=json.loads(line);row['metadata']['raw_locations']=[f'{shard.name}:{line_no}'];raw.append(row)
    raw_counts=Counter(s for r in raw for s in r['metadata']['training_sources'])
    unique={}
    for row in raw:
        key=row['metadata']['record_id']
        if key not in unique:unique[key]=row
        else:
            old=unique[key]['metadata'];new=row['metadata']
            old['training_sources']=sorted(set(old['training_sources'])|set(new['training_sources']))
            old['raw_locations'].extend(new['raw_locations'])
    rows=list(unique.values());patterns=defaultdict(list)
    for row in rows:
        meta=row['metadata'];meta['rendering_id']=meta['scenario_group_id']
        meta['scenario_group_id']='toy_eligibility_join_v1'
        meta['split_status']='single_base_world_no_independent_holdout'
        pattern=canonical_pattern(row);meta['canonical_pattern_id']=sha256(pattern.encode()).hexdigest()
        patterns[meta['canonical_pattern_id']].append(row)
        meta['assembled_record_sha256']=sha256(json.dumps(row,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    write_jsonl=lambda path,data:path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in data),encoding='utf-8')
    write_jsonl(folder/'preferences.jsonl',rows)
    slim=[]
    for pattern,group in patterns.items():
        r=group[0];members=sorted(set(s for x in group for s in x['metadata']['training_sources']))
        slim.append({k:r[k] for k in ('prompt','chosen','rejected')}|{'metadata':{
            'canonical_pattern_id':pattern,'training_sources':members,'multiplicity':len(group),
            'record_ids':[x['metadata']['record_id'] for x in group],
            'scenario_group_id':'toy_eligibility_join_v1','rule_family_id':'identity_attribute_join_v1',
            'world_config':r['metadata']['world_config'],'replay_prefix':r['metadata']['replay_prefix'],
            'evidence_class':'synthetic_structural_proxy_only','split_status':'no_independent_holdout'}})
    write_jsonl(folder/'canonical_preferences.jsonl',slim)
    sources={s:{r['metadata']['record_id'] for r in rows if s in r['metadata']['training_sources']} for s in ('ci','future','repair')}
    conflicts=[];directions=set()
    for row in rows:
        key=(row['prompt'],row['chosen'],row['rejected'])
        if (key[0],key[2],key[1]) in directions:conflicts.append(row['metadata']['record_id'])
        directions.add(key)
    summary={'raw_records':len(raw),'replayed_records':verified,'unique_surface_pairs':len(rows),
             'canonical_patterns':len(patterns),'surface_renderings':len({r['metadata']['rendering_id'] for r in rows}),
             'independent_base_worlds':1,'rule_families':1,'eligible_answers':['yes'],
             'raw_source_counts':dict(raw_counts),
             'unique_source_counts':{s:len(v) for s,v in sources.items()},
             'source_overlaps':{a+'_'+b:len(sources[a]&sources[b]) for a,b in (('ci','future'),('ci','repair'),('future','repair'))},
             'contradictory_identical_prompt_pairs':conflicts,
             'primary_split':{'train_base_worlds':1,'validation_base_worlds':0,'test_base_worlds':0},
             'generalization_evaluation_ready':False,'llm_training_ready':False,
             'reason':'All renderings share one eligible-yes relation/task template; no independent evaluation split exists.'}
    write_result(folder/'dataset_summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');assemble(p.parse_args().folder)
