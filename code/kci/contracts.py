"""KCI-Align v0.5: policy-independent typed action, evidence, and data contracts.

The prototype uses a finite, auditable synthetic world. It is *not* a semantic
LLM privacy judge or an implementation of arbitrary natural-language CI norms.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from copy import deepcopy
from enum import Enum
from typing import Any, Protocol, Mapping, Sequence
import json

class Verdict(str, Enum):
    VALID = 'valid'
    VIOLATION = 'violation'
    UNKNOWN = 'unknown'
    INVALID_ACTION = 'invalid_action'

class RefStatus(str, Enum):
    EXACT = 'exact_within_horizon'
    PROVEN_UNREACHABLE = 'proven_unreachable_within_horizon'
    WITNESS_ONLY = 'verified_witness_only'
    UNKNOWN = 'unknown'

@dataclass(frozen=True)
class Fact:
    id: str
    predicate: str
    subject: str
    value: str
    text: str

@dataclass(frozen=True)
class Message:
    event_id: str
    sender: str
    recipient: str
    purpose: str
    fact_ids: tuple[str,...]
    body: str
    delivered: bool = True

@dataclass(frozen=True)
class Action:
    actor: str
    kind: str
    recipient: str = ''
    fact_ids: tuple[str,...] = ()
    body: str = ''
    answer: str = ''
    def key(self) -> tuple:
        return self.actor, self.kind, self.recipient, self.fact_ids, self.body, self.answer
    def as_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
    @classmethod
    def from_mapping(cls, data: Mapping[str,Any]) -> 'Action':
        if not isinstance(data,Mapping):raise ValueError('Action must be a JSON object')
        keys={'actor','kind','recipient','fact_ids','body','answer'}
        if not set(data).issubset(keys):
            raise ValueError(f'Unexpected action keys {sorted(set(data)-keys)}')
        if not isinstance(data.get('actor'),str) or not isinstance(data.get('kind'),str):
            raise ValueError('actor/kind must be strings')
        fids=data.get('fact_ids',[])
        if not isinstance(fids,(tuple,list)) or any(not isinstance(i,str) for i in fids):
            raise ValueError('fact_ids must be a list of strings')
        return cls(actor=data['actor'],kind=data['kind'],recipient=str(data.get('recipient','')),
                   fact_ids=tuple(fids),body=str(data.get('body','')),answer=str(data.get('answer','')))

@dataclass(frozen=True)
class Observation:
    scenario_group_id: str
    norm_version: str
    oracle_version: str
    actor: str
    task: str
    purpose: str
    turn: int
    remaining_steps: int
    local_facts: tuple[Fact,...]
    # The trusted receipt log is scoped by read permissions; not the raw world state.
    authorized_receipts: tuple[Message,...]
    visible_ci_norms: tuple[str,...]
    pending_request: bool
    awaiting_notification: bool
    def as_prompt(self) -> str:
        data={
            'role':self.actor,'task':self.task,'purpose':self.purpose,
            'turn':self.turn,'remaining_steps':self.remaining_steps,
            'local_facts':[{'id':f.id,'text':f.text} for f in self.local_facts],
            'authorized_delivered_messages':[asdict(m) for m in self.authorized_receipts],
            'visible_ci_norms':list(self.visible_ci_norms),
            'pending_request':self.pending_request,'awaiting_notification':self.awaiting_notification,
            'instruction':'Choose one next executable agent action. Return a JSON Action.'
        }
        return json.dumps(data,ensure_ascii=False,sort_keys=True)

@dataclass(frozen=True)
class CandidateBatch:
    actions: tuple[Action,...]
    exhaustive: bool
    generator_version: str
    rejected_parse_count: int = 0

class CandidateGenerator(Protocol):
    version: str
    def generate(self, observation:Observation, limit:int=8) -> CandidateBatch:
        """No World/Oracle/CI labels accepted; input is authorized observation only."""
        ...

@dataclass(frozen=True)
class StepAudit:
    verdict: Verdict
    direct_evidence: tuple[str,...] = ()
    inference_evidence: tuple[str,...] = ()
    new_knowledge: tuple[str,...] = ()
    reason: str = ''
    semantic_verifier: str = 'structured_exact_v1'

@dataclass
class State:
    actor: str
    scenario_id: str = "unassigned_scenario"
    norm_version: str = "unassigned_norm"
    oracle_version: str = "unassigned_oracle"
    turn: int = 0
    max_turns: int = 10
    receipts: list[Message] = field(default_factory=list)
    local_fact_ids: dict[str,set[str]] = field(default_factory=dict)
    pending_request: bool = False
    finalized: bool = False
    final_correct: bool = False
    done: bool = False
    success: bool = False
    any_violation: bool = False
    cost: float = 0.0
    final_answer: str = ''
    def signature(self)->tuple:
        return (self.scenario_id,self.norm_version,self.oracle_version,
                self.actor,self.turn,self.max_turns,
                tuple((m.event_id,m.sender,m.recipient,m.purpose,m.fact_ids,m.body,m.delivered)
                      for m in self.receipts),
                tuple(sorted((a,tuple(sorted(fids))) for a,fids in self.local_fact_ids.items())),
                self.pending_request,self.finalized,self.final_correct,self.done,self.success,
                self.any_violation,self.cost,self.final_answer)

@dataclass(frozen=True)
class TraceStep:
    before: State
    action: Action
    audit: StepAudit
    after: State

@dataclass(frozen=True)
class Trace:
    steps: tuple[TraceStep,...]
    # An already-terminal start has a valid zero-action continuation. Retain its
    # endpoint instead of conflating it with an empty, unobserved rollout.
    start_state: State | None = None
    def __post_init__(self):
        if self.start_state is not None:
            object.__setattr__(self,'start_state',deepcopy(self.start_state))
    def is_safe(self)->bool:
        return (not (self.start_state and self.start_state.any_violation)
                and all(st.audit.verdict==Verdict.VALID for st in self.steps))
    @property
    def final(self)->State:
        if not self.steps:
            if self.start_state is not None:return self.start_state
            raise ValueError('Trace contains no steps or observed start state')
        return self.steps[-1].after
    def actions(self)->tuple[str,...]:
        return tuple(f'{st.action.actor}:{st.action.kind}' for st in self.steps)

@dataclass(frozen=True)
class RefResult:
    status: RefStatus
    best_trace: Trace | None
    best_success_cost: float | None
    expanded_states: int
    reason: str
    verifier_version: str
    generator_version: str
    horizon: int
    search_budget: tuple[int,int] | None = None

@dataclass(frozen=True)
class CCEReport:
    state_id: str
    observation: Observation
    action: Action
    immediate: StepAudit
    reference: RefResult | None
    actual_traces: tuple[Trace,...]
    policy_version: str
    reference_version: str = ''
    generator_version: str = ''
    search_budget: tuple[int,int] | None = None
    horizon: int | None = None

@dataclass(frozen=True)
class Preference:
    observation: Observation
    chosen: Action
    rejected: Action
    source: str
    state_id: str
    evidence: dict[str,Any]
    def to_dpo_record(self)->dict[str,Any]:
        if self.chosen.actor!=self.observation.actor or self.rejected.actor!=self.observation.actor:
            raise ValueError('Preference actor mismatch')
        if self.chosen.key()==self.rejected.key():
            raise ValueError('Identical action preference')
        if not self.evidence.get('certified',False):
            raise ValueError('Uncertified preference must not become DPO training data')
        return {'prompt':self.observation.as_prompt(),'chosen':self.chosen.as_json(),
                'rejected':self.rejected.as_json(),'metadata':{'source':self.source,
                'state_id':self.state_id,
                'scenario_group_id':self.observation.scenario_group_id,
                'norm_version':self.observation.norm_version,
                'task_oracle_version':self.observation.oracle_version,
                **self.evidence}}

@dataclass(frozen=True)
class RepairResult:
    status: str
    preference: Preference | None
    reason: str
    failed_step: int | None = None
