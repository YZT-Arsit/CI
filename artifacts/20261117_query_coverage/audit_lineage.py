"""Verify checkpoint ancestry and reconstruct the old trace sampler's depth exposure."""
import collections
import json
import random
from probe_data import R,ROOT,read,rows,sha


def main():
    base=R.parent/'20261116_relation_solution';action=R.parent/'20261117_action_binding'
    oldplan=read(base/'language_models/plan.json')
    assert all(sha(ROOT/n)==h for n,h in oldplan['source_hashes'].items())
    locations=[(base/'language_models/trace_17','sealed.json'),(action/'query_pilot/independent','SEALED.json'),
        (action/'learned_evidence/evidence','CHECKPOINT_SEALED.json'),(action/'coverage_repair/class_balanced','CHECKPOINT_SEALED.json'),
        (R.parent/'20261117_teacher_binding/repair/teacher_sft','CHECKPOINT_SEALED.json'),
        (R.parent/'20261117_optimization_repair/repair/continue_lower_lr','CHECKPOINT_SEALED.json'),
        (R.parent/'20261117_fallback_order/repair/interleaved','CHECKPOINT_SEALED.json')]
    seals=[read(p/n) for p,n in locations]
    for (p,n),s in zip(locations,seals):assert all(sha(p/'adapter'/f)==h for f,h in s['adapter_hashes'].items())
    incoming=[read(action/'query_pilot/PLAN.json')['initial_hashes'],read(action/'learned_evidence/PLAN.json')['initial_adapter_hashes'],
        read(action/'coverage_repair/PLAN.json')['initial_adapter_hashes'],read(R.parent/'20261117_teacher_binding/PLAN.json')['initial_adapter_hashes'],
        seals[5]['initial_adapter_hashes'],seals[6]['initial_adapter_hashes']]
    assert all(h==s['adapter_hashes'] for h,s in zip(incoming,seals[:-1]))
    assert read(R/'PLAN.json')['checkpoint']['seal']==seals[-1]
    indexed={};files=sorted((base/'dataset/train').glob('states_*.jsonl'))
    for p in files:
        for row in rows(p):
            obs=row['observation'];indexed.setdefault((row['base_group_id'],obs['role'],obs['committed_ticket']),[]).append(row)
    buckets={}
    for pair in indexed.values():
        assert len(pair)==2
        if all(len(r['optimal_indices'])==len(r['candidates']) for r in pair):continue
        key=(pair[0]['observation']['role'],any(not c['immediate_safe'] for r in pair for c in r['candidates']),any(c['safe_completion'] for r in pair for c in r['candidates']))
        buckets.setdefault(key,[]).append(sorted(pair,key=lambda r:r['world_id']))
    rng=random.Random(17);keys=sorted(buckets);counts=collections.Counter()
    for step in range(600):
        pair=rng.choice(buckets[rng.choice(keys)]);rng.randrange(10**9)
        depth=len(pair[0]['observation']['contract']['providers'][0]['links']);counts[depth]+=1
        # response() consumes exactly one choice per side, whose RNG consumption depends only on optimal-set size.
        for row in pair:rng.choice(row['optimal_indices'])
    assert set(counts)=={1,2,3} and len(read(base/'language_models/trace_17/training.json'))==600
    result={'verified_checkpoint_chain':[p.relative_to(ROOT).as_posix() for p,n in locations],
        'checkpoint_weight_hashes':[s['adapter_hashes']['adapter_model.safetensors'] for s in seals],
        'old_source_hashes_verified':oldplan['source_hashes'],'current_train_file_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'reconstructed_trace17_pair_counts_by_depth':dict(counts),'trace_training_steps':600,
        'qualification':'Deterministic sampler reconstruction from frozen source and current preserved TRAIN; original per-step sampled row IDs were not logged. Eligibility alone already rules out calling depth3 an unseen training family.',
        'scope_correction':'Depth2/3 coverage is restored in recent query/action continuation, not introduced for the first time in the full experimental ancestry.',
        'model_calls':0,'errors':0}
    (R/'LINEAGE_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['current_train_file_hashes','old_source_hashes_verified','checkpoint_weight_hashes']}))


if __name__=='__main__':main()
