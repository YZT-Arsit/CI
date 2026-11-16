"""Predeclared finite comparison/look budgets around the existing paired test.

This is union-bound accounting, not an anytime-valid method. Guarantees remain
conditional on the paired comparator's assumptions and on using one ledger for
the whole declared campaign. Copying a plan into independent ledgers is not a
new statistical budget. No evidence is recovered by refunding failed attempts.
"""
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path

from .continuation import ContinuationEvaluator
from .paired import PairedReferenceComparator


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                            separators=(',', ':')).encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class ComparisonIdentity:
    pair_id: str
    state_id: str
    prompt_sha256: str
    action_a_sha256: str
    action_b_sha256: str
    environment_version: str
    reference_policy_version: str
    horizon: int
    seed_base: int

    @classmethod
    def from_context(cls, pair_id, env, policy, state, action_a, action_b, seed_base=71):
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError('pair_id must be nonempty')
        if type(seed_base) is not int or seed_base < 0:
            raise ValueError('seed_base must be a nonnegative integer')
        if action_a.key() == action_b.key() or action_a.actor != state.actor or action_b.actor != state.actor:
            raise ValueError('distinct actions at the same acting state are required')
        return cls(pair_id, ContinuationEvaluator.state_id(state),
                   _hash(env.observation(state, state.actor).as_prompt()),
                   _hash(action_a.as_json()), _hash(action_b.as_json()),
                   env.version, policy.version, state.max_turns, seed_base)


@dataclass(frozen=True)
class EvidenceBudgetPlan:
    comparisons: tuple[ComparisonIdentity, ...]
    sample_sizes: tuple[int, ...]
    total_delta: float = .05

    def __post_init__(self):
        if (not self.comparisons or len({c.pair_id for c in self.comparisons}) != len(self.comparisons)):
            raise ValueError('predeclare distinct comparison IDs')
        if (not self.sample_sizes or any(type(n) is not int or n < 2 for n in self.sample_sizes)
                or tuple(sorted(set(self.sample_sizes))) != self.sample_sizes):
            raise ValueError('predeclare strictly increasing sample sizes >=2')
        if not isfinite(self.total_delta) or not 0 < self.total_delta < 1:
            raise ValueError('total_delta must be in (0,1)')

    @property
    def delta_per_look(self):
        return self.total_delta / (len(self.comparisons) * len(self.sample_sizes))

    @property
    def plan_id(self):
        return _hash(asdict(self))

    def as_dict(self):
        return {'plan_id': self.plan_id, **asdict(self), 'delta_per_look': self.delta_per_look,
                'method': 'fixed_finite_union_bound_NOT_anytime_valid',
                'maximum_claims': len(self.comparisons) * len(self.sample_sizes)}


class BudgetedPairedComparator:
    """Local append-only ledger; reserve before observing a comparison outcome."""
    def __init__(self, plan: EvidenceBudgetPlan, ledger_path):
        self.plan = plan
        self.path = Path(ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix + '.lock')
        with self._locked():
            if not self.path.exists():
                self._append({'event': 'plan', **plan.as_dict()})
            self._read()

    @contextmanager
    def _locked(self):
        fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            yield
        finally:
            os.close(fd)
            self.lock.unlink()

    def _append(self, row):
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _read(self):
        rows = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines()]
        if not rows or rows[0] != json.loads(json.dumps({'event': 'plan', **self.plan.as_dict()})):
            raise ValueError('ledger has a different or modified preregistered plan')
        expected = {c.pair_id for c in self.plan.comparisons}
        reserved = set()
        finished = set()
        for row in rows[1:]:
            if row.get('plan_id') != self.plan.plan_id:
                raise ValueError('ledger plan identity mismatch')
            key = (row.get('pair_id'), row.get('samples'))
            if key[0] not in expected or key[1] not in self.plan.sample_sizes:
                raise ValueError('unregistered ledger claim')
            if row['event'] == 'reserved':
                if key in reserved or row.get('delta') != self.plan.delta_per_look:
                    raise ValueError('duplicate or incorrect ledger reservation')
                reserved.add(key)
            elif row['event'] in {'completed', 'failed'}:
                if key not in reserved or key in finished:
                    raise ValueError('ledger completion without a unique reservation')
                finished.add(key)
            else:
                raise ValueError('unknown ledger event')
        return rows, reserved

    def compare(self, pair_id, samples, env, policy, state, action_a, action_b):
        matches = [c for c in self.plan.comparisons if c.pair_id == pair_id]
        if len(matches) != 1 or samples not in self.plan.sample_sizes:
            raise ValueError('comparison/look not preregistered')
        declared = matches[0]
        actual = ComparisonIdentity.from_context(pair_id, env, policy, state, action_a, action_b,
                                                 declared.seed_base)
        if actual != declared:
            raise ValueError('state/prompt/action/environment/policy identity changed')
        with self._locked():
            _, reserved = self._read()
            if (pair_id, samples) in reserved:
                raise ValueError('this comparison/look was already consumed, including failures')
            common = {'plan_id': self.plan.plan_id, 'pair_id': pair_id, 'samples': samples}
            self._append({'event': 'reserved', **common, 'delta': self.plan.delta_per_look})
            try:
                result = PairedReferenceComparator(env, policy, n=samples,
                         delta=self.plan.delta_per_look, seed_base=declared.seed_base).compare(state, action_a, action_b)
            except Exception as error:
                self._append({'event': 'failed', **common, 'error_type': type(error).__name__})
                raise
            if result.preference is not None:
                result.preference.evidence.update({
                    'batch_evidence_plan_id': self.plan.plan_id,
                    'batch_delta': self.plan.total_delta,
                    'batch_maximum_comparisons': len(self.plan.comparisons),
                    'batch_declared_looks': list(self.plan.sample_sizes),
                    'batch_pair_id': pair_id,
                })
            self._append({'event': 'completed', **common, 'status': result.status,
                          'preference': result.preference.to_dpo_record() if result.preference else None,
                          'report_sha256': _hash(asdict(result))})
            return result

    def accounting(self):
        with self._locked():
            rows, reserved = self._read()
        return {'plan_id': self.plan.plan_id, 'reserved_looks': len(reserved),
                'completed_looks': sum(r['event'] == 'completed' for r in rows),
                'spent_delta': len(reserved) * self.plan.delta_per_look,
                'maximum_delta': self.plan.total_delta,
                'scope': 'one declared campaign and its one persistent local ledger'}
