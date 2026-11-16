"""Recheck the frozen diagnostic evidence; no fitting or label changes."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
MATCHED='--matched' in sys.argv
CASES='cases_length_matched' if MATCHED else 'cases'
MODELS='models_length_matched' if MATCHED else 'models'
SUFFIX='_length_matched' if MATCHED else ''
sys.path.insert(0,str(ROOT/'code'))
from experiments.relation_gate_probe import rename
from kci.training_data import PinnedTokenizerAssets,TrainingSequenceAdapter


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def key(r):return r['kind'],r['horizon'],r['route_flip'],r['history_related_original']


def metric(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['map_index'],r['kind'],r['horizon'],r['route_flip'])].append(r['correct'])
    assert all(len(x)==2 for x in groups.values())
    return {'correct':sum(x['correct'] for x in rows),'total':len(rows),
            'both_correct':sum(all(x) for x in groups.values()),'groups':len(groups)}


def main():
    frozen=read(R/'input_hashes.json')
    assert all(sha256((ROOT/p).read_bytes()).hexdigest()==v for p,v in frozen.items())
    base={key(x):x for x in read(R/CASES/'map_00.json')['rows']}
    adapter=TrainingSequenceAdapter(PinnedTokenizerAssets.from_manifest(
        ROOT/'artifacts/20261112_cpu_pilot/tokenization/config.json'))
    token_counts=[];state_comparisons=0;oracle_correct=0
    for p in sorted((R/CASES).glob('map_*.json')):
        data=read(p);mapping=dict(data['map']);groups=defaultdict(list)
        for row in data['rows']:
            source=base[key(row)]
            assert row['values']==source['values'] and row['chosen']==source['chosen']
            assert json.loads(row['prompt'])==rename(json.loads(source['prompt']),mapping)
            assert row['traces']==rename(source['traces'],mapping)
            state_comparisons+=sum(len(t) for t in row['traces'].values())
            # A visible equality diagnostic, NOT learned or a method result.
            obs=json.loads(row['prompt']);history=obs['authorized_receipts'][0]['content']
            route=obs['authorized_routing_metadata']
            if row['kind']=='meeting':
                match=history['case']==route['upcoming_case_link']['case'];a=('SEND_ROSTER','PRIVATE_BOOKING')
            else:
                match=bool({x['company_id'] for x in history['rows']} & {x['company_id'] for x in route['owner_index']})
                a=('OPEN_INDEX','OPEN_VAULT')
            assert match==row['relation_matches']
            oracle_correct+=a[bool(match)]==row['chosen']
            n=len(adapter.tokenizer.encode(adapter.render(row['prompt']),add_special_tokens=False).ids)
            groups[(row['kind'],row['horizon'],row['route_flip'])].append(n)
        for group,counts in groups.items():
            token_counts.append({'map_index':data['map_index'],'group':group,'counts':counts,'equal':counts[0]==counts[1]})
    # The two related/disjoint members have equal tokenizer lengths everywhere.
    if MATCHED:assert all(x['equal'] for x in token_counts)
    write(R/f'token_length_checks{SUFFIX}.json',token_counts)
    model_metrics={};all_history=[]
    for p in sorted((R/MODELS).glob('*.json')):
        d=read(p);rows=d['rows'];assert len(rows)==544
        original={key(x):x for x in rows if x['map_index']==0}
        selections={'original':lambda x:x['map_index']==0 and not x['route_flip'],
            'symbol_swap':lambda x:x['map_index']==1 and not x['route_flip'],
            'fresh_symbols':lambda x:x['map_index']>=2 and not x['route_flip'],
            'fresh_factorial':lambda x:x['map_index']>=2}
        metrics={name:metric([x for x in rows if pred(x)]) for name,pred in selections.items()}
        fresh=[x for x in rows if x['map_index']>=2]
        same=sum(x['prediction']==original[key(x)]['prediction'] for x in fresh)
        metrics['fresh_prediction_agreement']={'same':same,'total':len(fresh)}
        gate=all(metrics[n]['correct']/metrics[n]['total']>=.95 and
                 metrics[n]['both_correct']/metrics[n]['groups']>=.90 for n in ('fresh_symbols','fresh_factorial'))
        metrics['predeclared_model_gate_passed']=gate and same/len(fresh)>=.95
        model_metrics[d['model']]=metrics
        if d['model'].startswith('history_'):all_history.extend(rows)
        # Recover the original stage-3 8 decisions exactly, not only its aggregate.
        old=read(ROOT/'artifacts/20261113_stage3/finite_v2'/d['model']/'result.json')
        for row in rows:
            if row['map_index']!=0 or row['route_flip']:continue
            case=f"{row['kind']}_h{int(row['history_related_original'])}_a0_t{row['horizon']}"
            previous=next(x for x in old['history_pair_evaluations'] if x['case_id']==case)
            assert row['correct']==previous['correct']
            assert abs(row['margin']-previous['margin'])<1e-12
    # Summing across seeds is descriptive only: maps and source tasks are shared.
    aggregate={}
    for name,pred in selections.items():
        ms=[v[name] for k,v in model_metrics.items() if k.startswith('history_')]
        aggregate[name]={f:sum(m[f] for m in ms) for f in ('correct','total','both_correct','groups')}
    dependency=read(R/'dependency.json');controls=[x for x in dependency if x['intervention']=='identity']
    cuts=[x for x in dependency if x['intervention']!='identity']
    assert len(controls)==8 and all(x['status']=='SAFE_SUCCESS' for x in controls)
    causal=read(R/'causal.json')
    assert all(x['verdict']==('VIOLATION' if x['intervention']=='identity' else 'VALID') for x in causal)
    costs={}
    for kind in ('meeting','query'):
        rows=read(R/f'cost_{kind}.json')['rows'];groups=defaultdict(list)
        for x in rows:groups[(x['private_overhead_weight'],x['sanitize_overhead_weight'])].append(x)
        strict=0
        for (private,sanitize),pair in groups.items():
            flip=(len(pair[0]['optimal_actions'])==len(pair[1]['optimal_actions'])==1 and pair[0]['optimal_actions']!=pair[1]['optimal_actions'])
            assert flip==(0<private<2*sanitize)
            strict+=flip
            for row in pair:
                common=5 if kind=='meeting' else 6
                actual=list(row['values'].values())
                assert actual==[common+2*sanitize*int(row['related']),common+private]
        costs[kind]={'weight_settings':len(groups),'strict_history_flips':strict,
            'both_safely_feasible_at_horizon20':all(x['both_routes_safely_feasible'] for x in rows),
            'necessary_and_sufficient_flip_condition_in_this_grid':'0 < private_weight < 2 * sanitize_weight',
            'unit_cost':groups[(1,1)],'zero_overhead':groups[(0,0)]}
    receipts=[]
    for name in (('matched_generation_jobs','matched_model_jobs') if MATCHED else ('generation_jobs','model_jobs')):
        receipts.extend(json.loads(x) for x in (R/f'{name}_receipts.jsonl').read_text().splitlines())
    assert all(x['status']=='completed' and x['source_hashes_unchanged'] and x['seconds']<=9.5 for x in receipts)
    summary={'baseline_commit':'507ddb769d75ecf2b83b30a59987532abf6c7e8f',
      'frozen_inputs_unchanged':True,'maps':34,'fresh_maps':32,'evaluated_states':544,
      'alpha_equivalent_state_trace_checks':state_comparisons,'matched_prompt_token_groups':len(token_counts),
      'all_within_pair_prompt_tokens_equal':all(x['equal'] for x in token_counts),
      'token_unequal_groups':sum(not x['equal'] for x in token_counts),'model_evaluations':12*544,
      'visible_relational_oracle_correct':oracle_correct,'oracle_is_learned':False,
      'original_model_margins_reproduced':True,'model_metrics':model_metrics,'history_aggregate':aggregate,
      'dependency_controls':len(controls),'dependency_interventions':len(cuts),
      'dependency_status_counts':dict(Counter(x['status'] for x in cuts)),
      'dependency_scope':'Fault injection with phase/global cache retained; proves message ledger is not the sole enforced dependency, not a reachable-state attack claim',
      'causal_interventions':len(causal),'causal_expected_outcomes_all_match':True,'cost_sensitivity':costs,
      'completed_workers':len(receipts),'max_worker_seconds':max(x['seconds'] for x in receipts),
      'GPU_training':'NO_GO','learned_relation_claim':'FALSIFIED_FOR_CURRENT_FINITE_CHECKPOINTS',
      'source_normative_validity':'NOT_ESTABLISHED','unseen_structure_generalization':'NOT_ESTABLISHED',
      'true_LLM_learning':'NOT_TESTED','all_evidence_consistency_checks_passed':True}
    write(R/f'summary{SUFFIX}.json',summary)
    print(json.dumps({k:summary[k] for k in ('history_aggregate','dependency_status_counts','GPU_training','alpha_equivalent_state_trace_checks','matched_prompt_token_groups')}))


if __name__=='__main__':main()
