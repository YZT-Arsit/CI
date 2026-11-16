"""Classify legacy future pairs by an explicit authorization intervention.

This intervention removes the synthetic derived-fact prohibition, not task
requirements. It distinguishes CI-mediated value from ordinary task value;
it alone does NOT prove a history-conditioned preference flip.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from kci.contracts import Action, Verdict
from kci.continuation import ContinuationEvaluator, ReferenceSearch, SearchBudget, certify_preference
from kci.multi_env import build_case
from kci.multi_candidates import MultiRuleCandidateGenerator


def worker(source, output):
    raw=json.loads(Path(source).read_text(encoding='utf-8'));spec=raw['case_spec']
    env=build_case(spec['family'],spec['answer'],spec['variant'],spec['task_op'],
                   allow_protected=True,max_turns=spec['max_turns'])
    cce=ContinuationEvaluator(env,ReferenceSearch(env,MultiRuleCandidateGenerator(),SearchBudget(3000,12)))
    reports={};states={};result=[]
    for row in raw['preferences']:
        m=row['metadata']
        if not m['source'].startswith('future'):continue
        key=json.dumps(m['replay_prefix'],sort_keys=True)
        if key not in states:
            st=env.initial()
            for value in m['replay_prefix']:
                st,audit=env.transition(st,Action.from_mapping(value))
                assert audit.verdict==Verdict.VALID
            states[key]=st
        state=states[key];pair=[]
        for side in ('chosen','rejected'):
            action=Action.from_mapping(json.loads(row[side]));k=(state.signature(),action.key())
            if k not in reports:reports[k]=cce.evaluate(state,action)
            pair.append(reports[k])
        relaxed=certify_preference(*pair)
        direction=('same' if relaxed and relaxed.chosen==pair[0].action else 'reversed' if relaxed else 'no_strict_preference')
        original_kind='task_reachability' if m['source']=='future_reachable' else 'compliant_path_cost'
        category=('ordinary_'+original_kind if direction=='same' else
                  'CI_mediated_'+original_kind if direction=='no_strict_preference' else 'REVIEW_REVERSED')
        result.append({'raw_record_id':m['raw_record_id'],'case_id':spec['case_id'],
            'source':m['source'],'category':category,'authorization_relaxed_direction':direction,
            'exact_relaxed_costs':[r.reference.best_success_cost if r.reference else None for r in pair],
            'exact_relaxed_status':[r.reference.status.value if r.reference else None for r in pair],
            'history_flip_proven':False})
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(result),'categories':dict(Counter(r['category'] for r in result))}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    worker(a.source,a.output)
