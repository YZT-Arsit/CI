"""Cost-free, source-adapted tasks with an explicit synthetic receipt obligation.

This contract does not claim that the source benchmarks impose these obligations.
Source computations are reused; knowledge rules and commitment semantics are new.
"""
from copy import deepcopy
from dataclasses import dataclass,field,asdict
from hashlib import sha256
from .dynamic_tasks import DynamicTask,TaskEvidenceError,canonical


@dataclass
class ReceiptState:
    phase: str='choose'
    ledger: list=field(default_factory=list)
    access_log: list=field(default_factory=list)
    consumption_log: list=field(default_factory=list)
    ticket: str=''
    done: bool=False
    success: bool=False
    violation: bool=False
    diagnostic_steps: int=0
    payload_bytes: int=0


RESOURCES={
    'meeting':(('HR','business_constraints'),('Room','public_schedule')),
    'query':(('Lina','company_gas_stations'),('Cassie','company_locations'),('JoinTool','company_id_map'))}


def closure(ledger):
    """Set-valued transitive joins, independent of row order and duplicate keys."""
    bodies=[m['content'] for m in ledger if m['recipient']=='R']
    identities={(a,b) for m in bodies for a,b in m.get('identities',[])}
    links={(a,b) for m in bodies for a,b in m.get('links',[])}
    histories={(a,b) for m in bodies for a,b in m.get('histories',[])}
    protected=set()
    for subject,ticket in identities:
        reachable={ticket}
        while True:
            expanded=reachable|{b for a,b in links if a in reachable}
            if expanded==reachable:break
            reachable=expanded
        protected|={(subject,value) for key,value in histories if key in reachable}
    return protected


class ReceiptTask:
    version='receipt_task_v1_synthetic_obligation'

    def __init__(self,world,source_root):
        self.world=deepcopy(world)
        self.kernel=DynamicTask(world['kind'],source_root,horizon=30)
        self.kernel.mailbox_acl['Worker']={'Worker'}
        self.expected=deepcopy(self.kernel.expected)
        self.binding={'version':self.version,'world':self.world,'source_hashes':self.kernel.source_hashes,
                      'source_acl':{k:sorted(v) for k,v in self.kernel.source_acl.items()},
                      'cost':0,'deadline':None,'source_native_obligation':False}

    def initial(self):
        s=ReceiptState()
        if self.world.get('history_delivered',True):
            self.kernel.send(s,'Archive','R',{'histories':self.world['histories']})
        return s

    def actor(self,s):return {'choose':'Planner','inputs':'Source','publish':'Worker'}.get(s.phase,'Terminal')

    def actions(self,s):
        if s.done:return ()
        if s.phase=='choose':return tuple('COMMIT:'+p['ticket'] for p in self.world['providers'])
        if s.phase=='inputs':return ('DELIVER_INPUTS','WITHHOLD')
        return ('PUBLISH_FULL','PUBLISH_MINIMAL','WITHHOLD')

    def provider(self,ticket):return next(p for p in self.world['providers'] if p['ticket']==ticket)

    def observation(self,s):
        # Binding declarations are authorized for the controller roles, NOT R's
        # background knowledge. Private source values and task truth are absent
        # from Planner input. Worker sees actual delivered computational inputs.
        actor=self.actor(s)
        receipts=[{k:deepcopy(m[k]) for k in ('event_id','sender','recipient','content')}
                  for m in s.ledger if m['recipient']=='R' or (actor=='Worker' and m['recipient']=='Worker')]
        return {'role':actor,'task_kind':self.world['kind'],'subject':self.world['subject'],
            'contract':{'providers':deepcopy(self.world['providers']),
                        'binding_required':self.world['binding_required'],
                        'publication_recipient':'R','original_commitment_must_be_closed':True,
                        'permission':self.world['allow'],'cost':0,'deadline':None},
            'committed_ticket':s.ticket,'delivered':receipts,'actions':list(self.actions(s))}

    def compute(self,s):
        values={};refs=[]
        for _,resource in RESOURCES[self.world['kind']]:
            data,ref=self.kernel.consume(s,'Worker',resource,'Worker')
            values[resource]=data['value'];refs.append(ref)
        if self.world['kind']=='meeting':
            result=self.kernel.solve_meeting(values['business_constraints'],values['public_schedule'])
        else:
            counts=self.kernel.table_rows({'rows':values['company_gas_stations']},'rows',{'company_id','gas_stations_count'})
            cities=self.kernel.table_rows({'rows':values['company_locations']},'rows',{'company_id','city'})
            names=self.kernel.table_rows({'rows':values['company_id_map']},'rows',{'company_id','company_name'})
            c={r['company_id']:r['gas_stations_count'] for r in counts}
            t={r['company_id']:r['city'] for r in cities}
            if len(c)!=len(counts) or len(t)!=len(cities):raise TaskEvidenceError('ambiguous join')
            if any(r['company_id'] not in c or r['company_id'] not in t for r in names):raise TaskEvidenceError('incomplete join')
            result=[{'company':r['company_name'],'stations':c[r['company_id']],'city':t[r['company_id']]} for r in names]
        return result,refs

    def transition(self,state,action):
        s=deepcopy(state)
        try:
            for message in s.ledger:self.kernel.verify_evidence(s,message['event_id'])
            if s.done:return s,{'verdict':'UNKNOWN','reason':'terminal'}
            if action in ('WAIT','REROUTE_COMPUTATION'):
                # These operations neither erase R's knowledge nor cancel the
                # original externally issued ticket. No step cost/deadline.
                s.diagnostic_steps+=1;return s,{'verdict':'VALID'}
            if action not in self.actions(s):raise TaskEvidenceError('outside declared action contract')
            before=closure(s.ledger);s.diagnostic_steps+=1
            if action=='WITHHOLD':s.done=True;s.phase='done'
            elif action.startswith('COMMIT:'):
                s.ticket=action.split(':',1)[1]
                self.kernel.send(s,'Planner','R',{'identities':[[self.world['subject'],s.ticket]]})
                s.phase='inputs'
            elif action=='DELIVER_INPUTS':
                for reader,resource in RESOURCES[self.world['kind']]:
                    value,ref=self.kernel.read_source(s,reader,resource)
                    self.kernel.send(s,reader,'Worker',{'type':resource,'value':value},inputs=(ref,))
                s.phase='publish'
            else:
                result,refs=self.compute(s)
                links=self.provider(s.ticket)['links'] if action=='PUBLISH_FULL' else []
                body={'type':'completed_receipt','ticket':s.ticket,'result':result,'links':links}
                self.kernel.send(s,'Worker','R',body,inputs=refs)
                # Independent task truth; never an execution-tool input.
                s.success=result==self.expected and (not self.world['binding_required'] or links==self.provider(s.ticket)['links'])
                s.done=True;s.phase='done'
            newly=closure(s.ledger)-before
            if newly and not self.world['allow']:s.violation=True;s.success=False;s.done=True;s.phase='done'
            return s,{'verdict':'VIOLATION' if newly and not self.world['allow'] else 'VALID',
                      'new_protected_relations':sorted(newly)}
        except TaskEvidenceError as error:
            return deepcopy(state),{'verdict':'UNKNOWN','reason':str(error)}

    def exact(self,s):
        """Finite semantic quotient. Arbitrary waits/reroutes are identity loops."""
        stack=[(deepcopy(s),[])];out=[]
        while stack:
            node,path=stack.pop()
            if node.done:
                out.append({'path':path,'ci':not node.violation,'success':node.success,
                            'cost':0,'final':asdict(node)})
                continue
            for a in self.actions(node):
                nxt,audit=self.transition(node,a)
                if audit['verdict']!='UNKNOWN':stack.append((nxt,path+[a]))
        if not out:return {'status':'UNKNOWN','value':None,'witness':None}
        key=lambda x:(int(x['ci']),int(x['ci'] and x['success']),0)
        best=max(out,key=key)
        return {'status':'EXACT_FINITE_SEMANTIC_QUOTIENT','value':key(best),'witness':best['path'],
                'safe_success':any(x['ci'] and x['success'] for x in out),'terminal_outcomes':out}

    def preference(self,s,a,b):
        values=[];audits=[]
        for action in (a,b):
            after,audit=self.transition(s,action);audits.append(audit)
            values.append(self.exact(after) if audit['verdict']!='UNKNOWN' else {'value':None})
        if any(v['value'] is None for v in values):return {'status':'UNKNOWN','chosen':None}
        va,vb=(v['value'] for v in values)
        chosen=0 if va>vb else 1 if vb>va else None
        # No strict training preference is emitted without a safe-success witness.
        if chosen is not None and not values[chosen].get('safe_success'):chosen=None
        return {'status':'PREFERENCE' if chosen is not None else 'TIE_OR_NO_SAFE_SUCCESS','chosen':chosen,
                'values':[va,vb],'immediate':[a['verdict'] for a in audits],
                'scope':'exact synthetic semantic contract, zero cost, no deadline'}
