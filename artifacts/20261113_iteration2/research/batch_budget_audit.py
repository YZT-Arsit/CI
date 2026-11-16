"""Independent local ledger failure-mode audit; no original artifacts are edited."""
from hashlib import sha256
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent / 'batch_budget_audit'
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code'))
from kci.batch_evidence import ComparisonIdentity, EvidenceBudgetPlan, BudgetedPairedComparator
from kci.continuation import DeterministicPolicy
from kci.toy_env import ToyCIEnv
from examples.run_demo import bad_current_policy


def main():
    OUT.mkdir(exist_ok=False)
    env = ToyCIEnv()
    state = env.initial()
    a = env.act('A', 'SKIP')
    b = env.act('A', 'SEND', 'R', ('identity',))
    policy = DeterministicPolicy(bad_current_policy, 'declared_version_v1')
    identity = ComparisonIdentity.from_context('pair', env, policy, state, a, b)
    plan = EvidenceBudgetPlan((identity,), (32, 64))
    runner = BudgetedPairedComparator(plan, OUT / 'version_scope.jsonl')
    first = runner.compare('pair', 32, env, policy, state, a, b)
    altered = DeterministicPolicy(lambda obs: env.act(obs.actor, 'ABSTAIN'), policy.version)
    same_id = ComparisonIdentity.from_context('pair', env, altered, state, a, b) == identity
    second = runner.compare('pair', 64, env, altered, state, a, b)

    def interrupted(obs):
        raise KeyboardInterrupt('controlled interruption')

    interrupted_policy = DeterministicPolicy(interrupted, 'interrupt_v1')
    interrupted_identity = ComparisonIdentity.from_context('pair', env, interrupted_policy, state, a, b)
    interrupted_plan = EvidenceBudgetPlan((interrupted_identity,), (32,))
    interrupted_path = OUT / 'interrupted.jsonl'
    try:
        BudgetedPairedComparator(interrupted_plan, interrupted_path).compare(
            'pair', 32, env, interrupted_policy, state, a, b)
    except KeyboardInterrupt:
        pass
    reopened = BudgetedPairedComparator(interrupted_plan, interrupted_path)
    blocked = False
    try:
        reopened.compare('pair', 32, env, interrupted_policy, state, a, b)
    except ValueError:
        blocked = True
    assert blocked and reopened.accounting()['reserved_looks'] == 1

    # Only this audit's own COPY is truncated; never touch a real campaign ledger.
    reset_path = OUT / 'manually_truncated_copy.jsonl'
    reset_path.write_text((OUT / 'version_scope.jsonl').read_text(encoding='utf-8').splitlines()[0] + '\n', encoding='utf-8')
    reset_count = BudgetedPairedComparator(plan, reset_path).accounting()['reserved_looks']
    result = {
        'scope': 'independent deterministic local wrapper audit, not model calibration',
        'same_declared_version_accepts_different_callable': same_id,
        'first_look_status': first.status,
        'different_callable_later_look_status': second.status,
        'interrupted_reservation_survives_reopen': blocked,
        'interrupted_accounting': reopened.accounting(),
        'manually_truncated_copy_reserved_looks': reset_count,
        'interpretation': {
            'version': 'Caller must bind truthful version to immutable code/weights; wrapper checks declared identity only.',
            'durability': 'Reserve-before-run survives normal reopening and an in-process BaseException.',
            'storage': 'The local API is append-only, not tamper-evident; copied/truncated files are outside its trusted-ledger guarantee.',
            'hard_crash': 'Not simulated; a stale lock can fail closed and needs explicit safe recovery.'
        },
        'source_sha256': {
            str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / 'code/kci/batch_evidence.py', ROOT / 'code/kci/paired.py']
        }
    }
    (OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
