"""Bounded local campaign integration; no claim of real-model calibration."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from math import ceil, log
from pathlib import Path
import time

from kci.batch_evidence import ComparisonIdentity, EvidenceBudgetPlan, BudgetedPairedComparator
from kci.continuation import DeterministicPolicy
from kci.toy_env import ToyCIEnv
from examples.run_demo import bad_current_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists(): raise FileExistsError('new campaign output directory required')
    out.mkdir(parents=True)
    started = time.perf_counter()
    policy = DeterministicPolicy(bad_current_policy, 'deterministic_bad_history_control_v1')
    environments = [ToyCIEnv(world_id=f'budget_fixture_{i}') for i in range(8)]
    identities = []
    for index, env in enumerate(environments):
        identities.append(ComparisonIdentity.from_context(str(index), env, policy, env.initial(),
                          env.act('A','SKIP'), env.act('A','SEND','R',('identity',)), 71+index*1000000))
    plan = EvidenceBudgetPlan(tuple(identities), (32,64,128), .05)
    (out/'plan.json').write_text(json.dumps(plan.as_dict(), indent=2), encoding='utf-8')
    runner = BudgetedPairedComparator(plan, out/'ledger.jsonl')
    results = []
    env = environments[0]
    for n in plan.sample_sizes:
        result = runner.compare('0', n, env, policy, env.initial(), env.act('A','SKIP'),
                                env.act('A','SEND','R',('identity',)))
        results.append(asdict(result))
    rejected = []
    runner = BudgetedPairedComparator(plan, out/'ledger.jsonl')
    for n in (64,256):
        try:
            runner.compare('0', n, env, policy, env.initial(), env.act('A','SKIP'),
                           env.act('A','SEND','R',('identity',)))
        except ValueError as error:
            rejected.append({'samples':n,'reason':str(error)})
    assert len(rejected)==2
    accounting = runner.accounting()
    assert accounting['reserved_looks']==3 and accounting['spent_delta']<=.05
    grid = []
    for comparisons in (1,8,128,2048):
        for looks in (1,4):
            delta=.05/(comparisons*looks)
            for gap in (.05,.15,.5):
                grid.append({'comparisons':comparisons,'looks':looks,'delta_per_look':delta,
                             'observed_gap':gap,'n_for_radius_strictly_below_gap':
                             int(2*log(4/delta)/(gap*gap))+1,
                             'scope':'algebraic threshold only; not power or sample-size guarantee'})
    (out/'reports.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    source = {str(p):sha256(p.read_bytes()).hexdigest() for p in
              [Path(__file__),Path('kci/batch_evidence.py'),Path('kci/paired.py')]}
    summary={'comparisons_preregistered':8,'looks_preregistered':[32,64,128],
             'delta_per_look':plan.delta_per_look,'accounting':accounting,
             'comparison_statuses':[r['status'] for r in results],
             'misuse_blocked':rejected,'budget_threshold_grid':grid,'source_sha256':source,
             'seconds':time.perf_counter()-started,
             'scope':'same deterministic toy fixture, distinct bookkeeping identities; not 8 independent tasks',
             'guarantee_conditions':['valid individual bounds','correct audits','frozen identity',
                                     'single ledger for whole campaign','predeclared finite looks'],
             'limitations':['not anytime valid','not real model error calibration',
                            'no universal protection against running an unauthorized second ledger']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({'seconds':summary['seconds'],'statuses':summary['comparison_statuses'],
                      'accounting':accounting}))


if __name__=='__main__': main()
