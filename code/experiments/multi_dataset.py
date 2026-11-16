"""Enumerate declared finite multi-rule states and retain replayable evidence.

Raw rows precede the independent observational-equivalence gate. More rows or
new numeric renderings are not more independent workflows.
"""
import argparse
from collections import Counter, deque
from dataclasses import asdict
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
import time

from kci.contracts import Action, Trace, TraceStep, Verdict
from kci.continuation import ContinuationEvaluator, ReferenceSearch, SearchBudget, certify_preference
from kci.datasets import export_verified_preferences
from kci.multi_env import FAMILIES, build_case
from kci.multi_candidates import MultiRuleCandidateGenerator
from kci.repair import DownstreamPolicyRepair


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), encoding='utf-8')


def case_env(spec):
    return build_case(spec['family'], spec['answer'], spec['variant'], spec['task_op'],
                      allow_protected=spec['allow_protected'], max_turns=spec['max_turns'])


def cases():
    rows=[]
    for family in FAMILIES:
        for op in (('membership',) if family=='set_intersection' else ('threshold','parity')):
            for answer in ('yes','no'):
                for permitted in (False,True):
                    for variant in (0,1):
                        for horizon in (5,9):
                            rows.append({'case_id':f'{family}_{op}_{answer}_{int(permitted)}_{variant}_{horizon}',
                                         'family':family,'task_op':op,'answer':answer,'variant':variant,
                                         'allow_protected':permitted,'max_turns':horizon})
    return rows


def build(spec, output):
    started=time.perf_counter();env=case_env(spec);generator=MultiRuleCandidateGenerator()
    reference=ReferenceSearch(env,generator,SearchBudget(3000,12))
    evaluator=ContinuationEvaluator(env,reference)
    frontier=deque([(env.initial(),[])]);seen=set();state_rows=[];preferences=[];failures=[]
    while frontier:
        state,prefix=frontier.popleft()
        if state.signature() in seen or state.done:continue
        seen.add(state.signature())
        if len(seen)>400:raise RuntimeError('predeclared state cap exceeded; do not silently call complete')
        observation=env.observation(state,state.actor)
        batch=generator.generate(observation,12)
        assert batch.exhaustive
        reports=[evaluator.evaluate(state,a) for a in batch.actions]
        action_rows=[]
        for action,report in zip(batch.actions,reports):
            after,audit=env.transition(state,action)
            ref=report.reference
            action_rows.append({'action':asdict(action),'immediate':audit.verdict.value,
                                'evidence':asdict(audit),'reference_status':ref.status.value if ref else None,
                                'minimum_safe_total_cost':ref.best_success_cost if ref else None,
                                'witness_actions':[asdict(s.action) for s in ref.best_trace.steps]
                                if ref and ref.best_trace else None})
            if audit.verdict==Verdict.VALID and not after.done:
                frontier.append((after,prefix+[asdict(action)]))
            if audit.verdict==Verdict.VIOLATION or (audit.verdict==Verdict.VALID and after.done and not after.success):
                failures.append(prefix+[asdict(action)])
        state_rows.append({'state_id':evaluator.state_id(state),'prompt':observation.as_prompt(),
                           'prefix':prefix,'actions':action_rows})
        for (ai,a),(bi,b) in combinations(enumerate(reports),2):
            pref=certify_preference(a,b)
            if pref is None:continue
            record=export_verified_preferences([pref])[0]
            metadata=record['metadata']
            metadata.update({'case_spec':spec,'replay_prefix':prefix,'world_configuration':env.configuration(),
                             'source_world_group_id':'multi_rule_handwritten_workflow_v1',
                             'workflow_group_id':env.workflow_group_id,'base_case_group_id':env.base_case_group_id,
                             'rule_family_id':env.family,'scenario_group_id':env.workflow_group_id,
                             'split':'train','split_status':'primary_shared_source_no_independent_holdout',
                             'human_verified':False,'natural_language_semantics_verified':False,
                             'norm_source':'manual_synthetic','evidence_class':'exact_declared_finite_synthetic',
                             'training_sources':['ci' if a.immediate.verdict!=b.immediate.verdict else 'future'],
                             'certification_reports':[action_rows[ai],action_rows[bi]]})
            metadata['raw_record_id']=sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            preferences.append(record)
    # Deterministically sample confirmed failed traces for DPR, preserving all
    # failures separately. This is a repair-coverage diagnostic, not efficacy.
    repair_rows=[];repairer=DownstreamPolicyRepair(evaluator,generator)
    for prefix in failures[:16]:
        state=env.initial();steps=[]
        for raw in prefix:
            action=Action.from_mapping(raw);after,audit=env.transition(state,action)
            steps.append(TraceStep(state,action,audit,after));state=after
        result=repairer.repair(Trace(tuple(steps)))
        repair_rows.append({'failed_actions':prefix,'status':result.status,'reason':result.reason,
                            'failed_step':result.failed_step,
                            'preference':result.preference.to_dpo_record() if result.preference else None})
    source_hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in sorted(Path('kci').glob('*.py'))}
    result={'case_spec':spec,'environment_version':env.version,'expected_answer':env.expected_answer,
            'safe_nonterminal_states':len(state_rows),'preferences':preferences,'states':state_rows,
            'confirmed_failure_prefixes':failures,'repair_diagnostics':repair_rows,
            'source_sha256':source_hashes,'seconds':time.perf_counter()-started,
            'enumeration_complete':True,'scope':'declared finite action grammar, one shared workflow'}
    write(output,result)
    print(json.dumps({'case':spec['case_id'],'states':len(state_rows),'pairs':len(preferences),
                      'repairs':Counter(r['status'] for r in repair_rows),'seconds':result['seconds']}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan');parser.add_argument('--case-id')
    parser.add_argument('--output',required=True);parser.add_argument('--make-plan',action='store_true')
    args=parser.parse_args()
    if args.make_plan:
        specs=cases();write(Path(args.output),{'cases':specs,'primary_workflows':1,
            'rule_families':list(FAMILIES),'primary_independent_holdout':False,
            'exploratory_family_folds':[{'heldout':f,'train':[x for x in FAMILIES if x!=f]} for f in FAMILIES],
            'state_cap_per_case':400,'reference_max_nodes':3000,'reference_max_actions':12})
        return
    plan=json.loads(Path(args.plan).read_text(encoding='utf-8'))
    spec=next(r for r in plan['cases'] if r['case_id']==args.case_id)
    output=Path(args.output)
    if output.exists():raise FileExistsError('preserve prior case result')
    build(spec,output)


if __name__=='__main__':main()
