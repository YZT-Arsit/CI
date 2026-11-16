"""Recompute policy means and replay recorded executions against repaired tools."""
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from collections import Counter
import argparse,gzip,json,sys

R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from experiments.task_evidence_dependency_probe import SOURCE
from kci.dynamic_tasks import DynamicTask,canonical
from kci.execution_envelope import VerifiedExecutionEnvelope,digest


def read(path):
    data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
    return json.loads(data.decode('utf-8'))


def audit(path,output):
    path=path.resolve()
    report=read(path);env=DynamicTask(report['kind'],SOURCE,related=report['related'],horizon=12)
    start=env.decision_start();binding=report['binding'];mode=binding['mode'];metric=binding['cost_metric']
    gateway=VerifiedExecutionEnvelope(env,start,env.options(start),binding['reference'],mode=mode,cost_metric=metric)
    assert canonical(gateway.binding)==canonical(binding)
    assert gateway.upper_certificate()==report['upper_certificate'] or canonical(gateway.upper_certificate())==canonical(report['upper_certificate'])
    records=report.get('execution_samples',report.get('executed_samples'))
    raw=Counter();replayed=0
    for record in records.values():
        for side,proof in enumerate(record['execution']):
            state=env.transition(start,proof['first_action'])[0]
            for step in proof['trace']:
                node=gateway.nodes[digest(asdict(state))]
                proposal=gateway.decode_proposal(step['raw_proposal'],state)
                if proposal is None:raw['invalid_draft']+=1
                else:
                    candidate,audit=env.transition(state,proposal)
                    raw['draft_'+audit['verdict']]+=1
                    if audit['verdict']=='VALID' and proposal not in node['eligible']:raw['safe_but_nonviable_draft']+=1
                if step['executed']=='GATEWAY_STOP':
                    assert not node['eligible'] and mode=='ci_only';state.done=True;state.success=False;continue
                expected=proposal if proposal in node['eligible'] else node['eligible'][0]
                assert step['executed']==expected
                state,audit=env.transition(state,step['executed'])
                assert audit==step['audit'] and digest(asdict(state))==step['after_state_sha256']
                replayed+=1
            assert asdict(state)==proof['final_state']
            assert not state.violation
            if mode=='safe_success':assert state.success
            expected_cost=state.turn if metric=='steps' else state.payload_bytes
            assert record['trials'][side]=={'ci':True,'success':state.success,'cost':expected_cost}
    exact=None
    if binding['reference'].get('name')=='seeded_proposal_control':
        memo={}
        def expectation(key):
            if key in memo:return memo[key]
            node=gateway.nodes[key];s=node['state'];cost=s.turn if metric=='steps' else s.payload_bytes
            if s.done or not node['eligible']:
                value=(Fraction(int(s.success and s.done)),Fraction(cost));memo[key]=value;return value
            choices=list(env.options(s))+['MALFORMED_ONE','MALFORMED_TWO']
            values=[]
            for action in choices:
                executed=action if action in node['eligible'] else node['eligible'][0]
                values.append(expectation(node['children'][executed]))
            value=tuple(sum(x[i] for x in values)/len(values) for i in (0,1));memo[key]=value;return value
        values=[expectation(digest(asdict(s))) for s in gateway.roots]
        ranks=[(v[0],-v[1]) for v in values]
        target=0 if ranks[0]>ranks[1] else 1 if ranks[1]>ranks[0] else None
        assert report['chosen'] is None or report['chosen']==target
        exact={'safe_success_probabilities':[str(v[0]) for v in values],
               'expected_total_costs':[str(v[1]) for v in values],
               'expected_total_costs_decimal':[float(v[1]) for v in values],
               'strict_reference_choice':target,
               'scope':'independent uniform per-step proposal idealization; not exact enumeration of the PRNG seed law or Qwen output distribution'}
    result={'file':str(path.relative_to(ROOT)),'executed_steps_replayed':replayed,
            'raw_draft_diagnostics':dict(raw),'exact_control_reference':exact,
            'chosen':report['chosen'],'status':report['status'],'all_passed':True}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('file','executed_steps_replayed','chosen','status','all_passed')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args();audit(Path(a.input),Path(a.output))
