"""Freeze the sampler-only repair, explicitly reusing development evidence."""
import collections
import json
import shutil
from paths import R,ROOT,PARENT,read,rows,sha,verify_prior
from contrast_data import schedule
from eval_worlds import check


def main():
    assert not (R/'PLAN.json').exists();verify_prior();parent=read(PARENT/'PLAN.json');batches=schedule()
    assert len(batches)==96 and sum(b['forced_contrast'] for b in batches)==54
    assert all(not check(c) for b in batches for pair in b['pairs'].values() for c in pair)
    (R/'train_batches.jsonl').write_text(''.join(json.dumps(b)+'\n' for b in batches),encoding='utf-8')
    names=['dev_worlds.json','query_observations.jsonl','query_labels.jsonl','fresh_observations.jsonl','fresh_labels.jsonl']
    for name in names:shutil.copy2(PARENT/name,R/name)
    deps=dict(parent['dependencies'])
    for name in [*parent['source_hashes'],*parent['data_hashes'],'PLAN.json','SEALED.json','RESULTS.json','PAIR_QUERY_RESULTS.json','CONTRAST_EXPOSURE.json','PROBE_SEALED.json']:
        deps[(PARENT/name).relative_to(ROOT).as_posix()]=sha(PARENT/name)
    plan={k:parent[k] for k in ['arms','evidence_arms','checkpoint','earlier_checkpoint','lr','seed','updates_per_arm','joint_batches','correction_batches',
        'update_guard_seconds','schedules','paired_conditions','probe_seal_sha256','query_count','episodes_per_arm']}
    plan.update(source_sha256=sha(R/'run.py'),dependencies=deps,
        source_hashes={n:sha(R/n) for n in ['paths.py','contrast_data.py','test_schedule.py','build.py','run.py','audit.py','audit_pair_queries.py']},
        data_hashes={n:sha(R/n) for n in names+['train_batches.jsonl','PLAN.md']},
        development_set_reused=True,parent_run=PARENT.relative_to(ROOT).as_posix(),
        reused_arms=['initial','earlier'],newly_trained_arms=['one_hop','mixed_depth'],whole_goal_complete=False,oracle_at_runtime=False)
    (R/'PLAN.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    pairplan={'main_plan_sha256':sha(R/'PLAN.json'),'source_sha256':sha(R/'audit_pair_queries.py'),
        'scope':'Prelaunch paired-query diagnostic on reused DEV, all label-changing native pairs. No new independent validation or primary-gate change.'}
    (R/'PAIR_QUERY_PLAN.json').write_text(json.dumps(pairplan,indent=2),encoding='utf-8')
    counts=collections.Counter();forced=collections.Counter()
    for b in batches:
        role=b['key'][0];d=b['mixed_depth'];a,c=b['pairs'][str(d)]
        counts[role,d]+=any(x['safe_completion']!=y['safe_completion'] for x,y in zip(a['candidates'],c['candidates']))
        if b['forced_contrast']:forced[role,d,b['factors'][0]]+=1
    assert len(counts)==9 and min(counts.values())>=6 and set(forced.values())=={3}
    receipt={'actual_reversal_batches':[{'role':r,'depth':d,'count':v} for (r,d),v in counts.items()],
        'forced_role_depth_kind_counts':{str(k):v for k,v in forced.items()},'forced_batches':54,'anchor_batches':42,
        'native_train_states_checked':576,'errors':0,'development_set_reused':True,'new_updates':246}
    (R/'PREFLIGHT.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))


if __name__=='__main__':main()
