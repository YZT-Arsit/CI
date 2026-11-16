"""Generate replayable original synthetic preferences, never third-party CI labels."""
import argparse
from collections import Counter,defaultdict
from dataclasses import asdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import time

from kci import ContinuationEvaluator, DownstreamPolicyRepair, ToyCandidateGenerator, certify_preference
from kci.contracts import Action, Verdict
from experiments.common import world, reference, prefix, policy, rollout, trace_evidence, write_result


def build(start,count):
    records=[];attempts=[]
    for index in range(start,start+count):
        for permitted,horizon,expensive in itertools.product((False,True),(6,9),(False,True)):
            costs={'REQUEST_RULES':2.0} if expensive else {}
            config={'index':index,'permitted':permitted,'horizon':horizon,'costs':costs}
            env=world(**config);cce=ContinuationEvaluator(env,reference(env));initial=env.initial()
            initial_actions=[env.act('A','SKIP'),env.act('A','SEND','R',('identity',)),env.act('A','SEND','R',('identity_other',))]

            def save(pair,subset,actions_before,reports,origin):
                if pair is None:return
                record=pair.to_dpo_record()
                key=sha256((record['prompt']+'\n'+record['chosen']+'\n'+record['rejected']).encode()).hexdigest()
                record['metadata'].update({'record_id':key,'training_sources':[subset],
                    'dataset':'kci_cpu_synthetic_v1','evidence_class':'synthetic_exact_bounded',
                    'rule_family_id':'identity_attribute_join_v1','world_config':config,
                    'replay_prefix':[asdict(a) for a in actions_before],
                    'origin':origin,'human_verified':False,'natural_language_semantics_verified':False,
                    'environment_version':env.version,
                    'supporting_reports':[{'action':asdict(r.action),'immediate':asdict(r.immediate),
                        'reference_status':r.reference.status.value if r.reference else None,
                        'reference_cost':r.reference.best_success_cost if r.reference else None,
                        'reference_witness':trace_evidence(r.reference.best_trace) if r.reference else None}
                        for r in reports]})
                records.append(record)

            reports=[cce.evaluate(initial,a) for a in initial_actions]
            for a,b in itertools.combinations(reports,2):
                pair=certify_preference(a,b)
                attempts.append({'world':index,'config':config,'kind':'future_A','pair':pair is not None,
                                 'statuses':[r.reference.status.value if r.reference else r.immediate.verdict.value for r in (a,b)]})
                save(pair,'future',[],(a,b),'same_prefix_current_legal_actions')
            for first in initial_actions:
                state,_=prefix(env,[first]);a=cce.evaluate(state,env.act('B','SEND','R',('band',)))
                b=cce.evaluate(state,env.act('B','REQUEST_RULES','R'));pair=certify_preference(a,b)
                subset='ci' if Verdict.VIOLATION in (a.immediate.verdict,b.immediate.verdict) else 'future'
                attempts.append({'world':index,'config':config,'kind':subset+'_B','pair':pair is not None,
                                 'statuses':[r.reference.status.value if r.reference else r.immediate.verdict.value for r in (a,b)]})
                save(pair,subset,[first],(a,b),'same_prefix_disclosure_vs_local_processing')
            for mode,first_kind in (('blind','SEND'),('wrong_answer','SKIP')):
                failed=rollout(env,policy(env,mode,first_kind))
                repair=DownstreamPolicyRepair(cce,ToyCandidateGenerator()).repair(failed)
                attempts.append({'world':index,'config':config,'kind':'repair_'+mode,'pair':repair.preference is not None,
                                 'status':repair.status})
                if repair.preference is not None:
                    k=repair.failed_step;state=failed.steps[k].before
                    rp=repair.preference
                    save(rp,'repair',[s.action for s in failed.steps[:k]],
                         (cce.evaluate(state,rp.chosen),cce.evaluate(state,rp.rejected)),
                         {'policy':mode,'failure_trace':trace_evidence(failed),'repair_index':k})
    return records,attempts


def verify_record(record):
    meta=record['metadata'];env=world(**meta['world_config'])
    st,trace=prefix(env,[Action.from_mapping(a) for a in meta['replay_prefix']])
    if not trace.is_safe():return 'unsafe_prefix'
    if env.observation(st,st.actor).as_prompt()!=record['prompt']:return 'prompt_mismatch'
    if ContinuationEvaluator.state_id(st)!=meta['state_id']:return 'state_mismatch'
    cce=ContinuationEvaluator(env,reference(env))
    chosen=Action.from_mapping(json.loads(record['chosen']));rejected=Action.from_mapping(json.loads(record['rejected']))
    pair=certify_preference(cce.evaluate(st,chosen),cce.evaluate(st,rejected))
    if pair is None or pair.chosen.key()!=chosen.key():return 'preference_not_reproduced'
    return 'verified'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=4)
    p.add_argument('--output',required=True);p.add_argument('--verify',action='store_true')
    args=p.parse_args();t=time.perf_counter();output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if args.verify:
        records=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        counts=Counter(verify_record(record) for record in records)
        core_root=Path(__file__).resolve().parents[1]/'kci'
        core_hashes={p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(core_root.glob('*.py'))}
        result={'records':len(records),'replay_results':dict(counts),'seconds':time.perf_counter()-t,
                'input_sha256':sha256(output.read_bytes()).hexdigest(),'verification_version':'replay_v2',
                'core_sha256':core_hashes,'verifier_script_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
        write_result(output.with_suffix('.verification.v2.json'),result);print({k:v for k,v in result.items() if k!='core_sha256'})
        raise SystemExit(0 if set(counts)<={'verified'} else 1)
    records,attempts=build(args.start,args.count)
    output.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in records),encoding='utf-8')
    summary={'start':args.start,'worlds':args.count,'records':len(records),'attempts':len(attempts),
             'by_source':dict(Counter(s for r in records for s in r['metadata']['training_sources'])),
             'seconds':time.perf_counter()-t,'attempt_results':attempts}
    write_result(output.with_suffix('.summary.json'),summary);print({k:v for k,v in summary.items() if k!='attempt_results'})
