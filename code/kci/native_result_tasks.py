"""Source-native result fields, with explicitly synthetic initial knowledge/CI.

No new completion-receipt field is required. The actual native task result is
computed and legally delivered internally before the disclosure decision.
"""
from copy import deepcopy
from .receipt_tasks import ReceiptTask,ReceiptState,RESOURCES,closure
from .ordering_receipt_tasks import OrderingReceiptTask
from .dynamic_tasks import TaskEvidenceError


def result_edges(kind,result):
    if kind=='meeting':return [['meeting:'+k,'room:'+v['room']] for k,v in sorted(result.items())]
    return [['company:'+r['company'],'city:'+r['city']] for r in result]


def native_closure(kind,ledger):
    interpreted=deepcopy(ledger)
    for m in interpreted:
        if m['recipient']=='R' and m['content'].get('type')=='native_result':
            m['content']['links']=result_edges(kind,m['content']['result'])
    return closure(interpreted)


class NativeResultTask(ReceiptTask):
    version='native_result_dependency_v1_synthetic_knowledge_norm'

    def __init__(self,world,source_root):
        cls=OrderingReceiptTask if world.get('kernel_variant')=='ordering' else ReceiptTask
        base=cls(world,source_root);self.__dict__.update(base.__dict__)
        self.binding.update(version=self.version,extra_receipt_requirement=False)

    def initial(self):
        s=super().initial()
        for reader,resource in RESOURCES[self.world['kind']]:
            value,ref=self.kernel.read_source(s,reader,resource)
            self.kernel.send(s,reader,'Worker',{'type':resource,'value':value},inputs=(ref,))
        result,refs=self.compute(s)
        for recipient in ('Planner','Worker'):
            self.kernel.send(s,'Worker',recipient,{'type':'computed_native_result','result':result},inputs=refs)
        return s

    def actions(self,s):
        if s.done:return ()
        if s.phase=='choose':return tuple('COMMIT:'+p['ticket'] for p in self.world['providers'])
        return ('PUBLISH_FULL','PUBLISH_MINIMAL','WITHHOLD')

    def observation(self,s):
        obs=super().observation(s)
        actor=self.actor(s)
        messages=[m for m in s.ledger if m['recipient']==actor and m['content'].get('type')=='computed_native_result']
        if len(messages)!=1:raise TaskEvidenceError('missing actual computed-result receipt')
        self.kernel.verify_evidence(s,messages[0]['event_id'])
        edges=result_edges(self.world['kind'],messages[0]['content']['result'])
        providers=[]
        for p in self.world['providers']:
            links=[e for e in edges if e[0]==p['ticket']]
            if len(links)!=1:raise TaskEvidenceError('candidate not bound to computed native output')
            providers.append({'ticket':p['ticket'],'links':links})
        obs['contract']['providers']=providers
        obs['contract']['original_commitment_must_be_closed']=False
        obs['contract']['native_full_result_required']=True
        obs['contract']['extra_receipt_requirement']=False
        obs['delivered']=[{k:deepcopy(m[k]) for k in ('event_id','sender','recipient','content')}
                          for m in s.ledger if m['recipient'] in ('R',actor)]
        return obs

    def transition(self,state,action):
        s=deepcopy(state)
        try:
            for m in s.ledger:self.kernel.verify_evidence(s,m['event_id'])
            if s.done or action not in self.actions(s):raise TaskEvidenceError('outside native-output action contract')
            before=native_closure(self.world['kind'],s.ledger);s.diagnostic_steps+=1
            if action.startswith('COMMIT:'):
                s.ticket=action.split(':',1)[1]
                self.kernel.send(s,'Planner','R',{'identities':[[self.world['subject'],s.ticket]]})
                s.phase='publish'
            elif action=='WITHHOLD':s.done=True;s.phase='done'
            else:
                body,ref=self.kernel.consume(s,'Worker','computed_native_result','Worker')
                result=body['result'] if action=='PUBLISH_FULL' else ({} if self.world['kind']=='meeting' else [])
                # No ticket/link field is added to the mandatory native result.
                self.kernel.send(s,'Worker','R',{'type':'native_result','result':result},inputs=(ref,))
                s.success=result==self.expected;s.done=True;s.phase='done'
            newly=native_closure(self.world['kind'],s.ledger)-before
            if newly and not self.world['allow']:s.violation=True;s.success=False;s.done=True;s.phase='done'
            return s,{'verdict':'VIOLATION' if newly and not self.world['allow'] else 'VALID','new_protected_relations':sorted(newly)}
        except TaskEvidenceError as error:return deepcopy(state),{'verdict':'UNKNOWN','reason':str(error)}
