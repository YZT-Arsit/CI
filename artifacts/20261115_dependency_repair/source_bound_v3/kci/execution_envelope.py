"""A verified executed-reference contract, not a proof about raw LLM outputs.

Finite tool transitions are verified once. Arbitrary proposals are routed through
that contract. No claim is made for free-form side effects outside this gateway.
"""
from copy import deepcopy
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
from pathlib import Path
from .dynamic_tasks import DynamicTask,canonical
from .contracts import Action
from .stratified import Trial


def digest(value):return sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class ExecutedPair:
    envelope_id: str
    seed: int
    trials: tuple
    evidence_id: str


class VerifiedExecutionEnvelope:
    """Certify CI, optionally success, for every proposal through this executor.

    ci_only: all accepted transitions are safe; failure/abort can still occur.
    safe_success: retain only viable safe transitions. Finite decreasing budget
    plus a viable fallback establishes success by induction, without enumerating
    the language model's output distribution. This changes the executed reference
    policy and must always be labelled as such.
    """
    version='verified_execution_envelope_v1'

    def __init__(self,env,start,actions,reference_identity,*,mode='ci_only',cost_metric='steps'):
        if type(env) is not DynamicTask:raise ValueError('only the audited DynamicTask implementation is supported')
        if mode not in ('ci_only','safe_success'):raise ValueError('unknown envelope mode')
        if cost_metric not in ('steps','payload_bytes'):raise ValueError('unknown cost metric')
        if not isinstance(reference_identity,dict) or not reference_identity:raise ValueError('frozen reference identity required')
        self.env=env;self.start=deepcopy(start);self.actions=tuple(actions)
        if len(self.actions)!=2 or self.actions[0]==self.actions[1]:raise ValueError('two distinct first actions required')
        self.mode=mode;self.cost_metric=cost_metric;self.reference_identity=deepcopy(reference_identity)
        self._environment=self.environment_fingerprint();self.nodes={};self.roots=[];self.samples={}
        if start.violation or start.done:raise ValueError('safe active initial state required')
        for message in start.ledger:env.verify_evidence(start,message['event_id'])
        if not env.allow and env.protected(start.ledger):raise ValueError('initial knowledge already violates the protected relation')
        for action in self.actions:
            after,audit=env.transition(start,action)
            if audit['verdict']!='VALID' or after.violation:raise ValueError('first action is not certified safe')
            self.roots.append(deepcopy(after));viable=self._visit(after)
            if mode=='safe_success' and not viable:raise ValueError('first branch has no verified safe completion')
        self.binding={'environment':self._environment,'start':digest(asdict(start)),
            'first_actions':self.actions,'reference':self.reference_identity,'mode':mode,
            'cost_metric':cost_metric,'executor_version':self.version}
        self.envelope_id=digest(self.binding)
        self.max_cost=max(self._cost(node['state']) for node in self.nodes.values())
        self.cost_ranges=tuple(self._cost_range(digest(asdict(state))) for state in self.roots)
        self._graph_hash=self.graph_hash()

    def environment_fingerprint(self):
        # Bind actual in-memory source/ACLs and implementation bytes; no model
        # or callback may mutate these between proposal and execution.
        if any(callable(x) for x in vars(self.env).values()):raise ValueError('instance method overrides are unsupported')
        root=Path(__file__).resolve().parent
        files={name:sha256((root/name).read_bytes()).hexdigest() for name in ('dynamic_tasks.py','execution_envelope.py')}
        def plain(x):
            if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
            if isinstance(x,set):return sorted(x)
            if isinstance(x,(list,tuple)):return [plain(v) for v in x]
            return x
        return digest({'environment':plain(vars(self.env)),'code':files})

    def graph_hash(self):
        return digest({key:{'eligible':node['eligible'],'viable':node['viable'],
                            'state':asdict(node['state']),'children':node['children']}
                       for key,node in self.nodes.items()})

    def verify_current(self,binding=None):
        if self.environment_fingerprint()!=self._environment:raise ValueError('environment/source/ACL/code drift')
        if self.graph_hash()!=self._graph_hash:raise ValueError('verified transition graph was modified')
        if digest(self.binding)!=self.envelope_id:raise ValueError('reference/execution binding changed')
        if binding is not None and canonical(binding)!=canonical(self.binding):raise ValueError('execution proof binding mismatch')

    def _cost(self,state):return state.turn if self.cost_metric=='steps' else state.payload_bytes

    def _cost_range(self,key):
        node=self.nodes[key]
        if node['state'].done or not node['eligible']:
            value=self._cost(node['state']);return value,value
        ranges=[self._cost_range(node['children'][a]) for a in node['eligible']]
        return min(x[0] for x in ranges),max(x[1] for x in ranges)

    def _visit(self,state):
        key=digest(asdict(state))
        if key in self.nodes:return self.nodes[key]['viable']
        children={};viable=bool(state.done and state.success and not state.violation)
        if not state.done:
            for action in self.env.options(state):
                after,audit=self.env.transition(state,action)
                if audit['verdict']!='VALID' or after.violation:continue
                if after.turn!=state.turn+1 or after.turn>self.env.horizon:raise ValueError('nondecreasing or invalid rank')
                child_viable=self._visit(after);children[action]=digest(asdict(after))
                viable=viable or child_viable
        eligible=tuple(sorted(a for a,k in children.items() if self.mode=='ci_only' or self.nodes[k]['viable']))
        self.nodes[key]={'state':deepcopy(state),'children':children,'eligible':eligible,'viable':viable}
        return viable

    def upper_certificate(self):
        self.verify_current()
        ranges=tuple(self._cost_range(digest(asdict(state))) for state in self.roots)
        if ranges!=self.cost_ranges:raise ValueError('cost support proof changed')
        return {'known':True,'ci_equal':True,'safe_success_equal':self.mode=='safe_success',
            'all_success':self.mode=='safe_success','envelope_id':self.envelope_id,
            'verified_transition_states':len(self.nodes),'scope':'executed_reference_through_verified_gateway_only',
            'raw_policy_risk_proven_zero':False,'reference_distribution_enumerated':False,
            'terminal_cost_ranges':ranges,
            'paired_cost_difference_bounds':(ranges[1][0]-ranges[0][1],ranges[1][1]-ranges[0][0]),
            'success_certificate_uses_exact_finite_viability':self.mode=='safe_success'}

    def decode_proposal(self,raw,state):
        if isinstance(raw,str) and raw.strip() in self.env.options(state):return raw.strip()
        if isinstance(raw,str):
            try:raw=json.loads(raw)
            except (ValueError,TypeError):return None
        if isinstance(raw,Action):
            raw=asdict(raw);raw['fact_ids']=list(raw['fact_ids'])
        fields={'actor','kind','recipient','fact_ids','body','answer'}
        if not isinstance(raw,dict) or set(raw)!=fields:return None
        if raw['actor']!=self.env.actor(state) or any(raw[x]!='' for x in ('recipient','body','answer')):return None
        if type(raw['fact_ids']) is not list or raw['fact_ids']!=[]:return None
        return raw['kind'] if isinstance(raw['kind'],str) and raw['kind'] in self.env.options(state) else None

    def execute(self,proposal,seed,side):
        state=deepcopy(self.roots[side]);trace=[];step=0
        while not state.done:
            key=digest(asdict(state));node=self.nodes[key]
            try:raw=proposal(self.env.observation(state),seed,step,side)
            except Exception as error:raw={'proposal_error':type(error).__name__}
            if self.environment_fingerprint()!=self._environment:raise ValueError('proposal mutated environment')
            action=self.decode_proposal(raw,state);eligible=node['eligible']
            if not eligible:
                if self.mode=='safe_success':raise AssertionError('viability invariant broken')
                state.done=True;state.success=False
                trace.append({'raw_proposal':repr(raw),'executed':'GATEWAY_STOP','blocked':True,'reason':'no safe continuation'})
                break
            executed=action if action in eligible else eligible[0]
            after,audit=self.env.transition(state,executed)
            if audit['verdict']!='VALID' or digest(asdict(after))!=node['children'][executed]:raise ValueError('transition differs from verified graph')
            trace.append({'raw_proposal':raw if isinstance(raw,(str,int,float,bool,list,dict)) or raw is None else repr(raw),
                'decoded_action':action,'eligible':eligible,'executed':executed,'blocked':executed!=action,
                'audit':audit,'after_state_sha256':digest(asdict(after))})
            state=after;step+=1
        if state.violation or (self.mode=='safe_success' and not state.success):raise AssertionError('executed contract violated')
        return Trial(True,state.success,self._cost(state)),{'first_action':self.actions[side],
            'trace':trace,'final_state':asdict(state),'raw_proposal_semantics':'unexecuted draft unless accepted'}

    def sample_pair(self,proposal,seed):
        if self.environment_fingerprint()!=self._environment:raise ValueError('environment drift')
        trials=[];proof=[]
        for side in (0,1):
            trial,trace=self.execute(proposal,seed,side);trials.append(trial);proof.append(trace)
        record={'envelope_id':self.envelope_id,'seed':seed,'trials':[asdict(x) for x in trials],'execution':proof}
        evidence_id=digest(record);self.samples[evidence_id]=record
        return ExecutedPair(self.envelope_id,seed,tuple(trials),evidence_id)

    def validate_sample(self,sample,seed):
        if not isinstance(sample,ExecutedPair) or sample.seed!=seed or sample.envelope_id!=self.envelope_id:
            raise ValueError('audited executed pair from this envelope/seed required')
        record=self.samples.get(sample.evidence_id)
        if record is None or digest(record)!=sample.evidence_id or record['trials']!=[asdict(x) for x in sample.trials]:
            raise ValueError('execution record missing or changed')
        return sample.trials
