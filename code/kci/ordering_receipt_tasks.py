"""Source-backed meeting-order constraints, outside the frozen legacy kernel."""
from copy import deepcopy
from .dynamic_tasks import DynamicTask,TaskEvidenceError
from .receipt_tasks import ReceiptTask


class OrderingMeetingKernel(DynamicTask):
    def business_constraints(self):
        fields={'cannot_use_room':('person','type','room_id'),
                'person_unavailable':('person','type','slot_id'),
                'meeting_ordering':('person','type','before_meeting_id','after_meeting_id')}
        rows=[]
        for c in self.utility['hidden_private_constraints']:
            if c['type'] not in fields:raise TaskEvidenceError('unsupported source constraint type')
            rows.append({k:c[k] for k in fields[c['type']]})
        return rows

    @staticmethod
    def validate_constraints(rows):
        if not isinstance(rows,list):raise TaskEvidenceError('typed constraint list required')
        for row in rows:
            if isinstance(row,dict) and row.get('type')=='meeting_ordering':
                if set(row)!={'person','type','before_meeting_id','after_meeting_id'} or any(not isinstance(v,str) or not v for v in row.values()):
                    raise TaskEvidenceError('invalid ordering constraint')
            else:DynamicTask.validate_constraints([row])

    @staticmethod
    def schedule_valid(result,constraints,public):
        basic=[c for c in constraints if c['type']!='meeting_ordering']
        if not DynamicTask.schedule_valid(result,basic,public):return False
        slots={s['id']:s for s in public['slots']}
        days={day:i for i,day in enumerate(('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))}
        for c in constraints:
            if c['type']!='meeting_ordering':continue
            if c['before_meeting_id'] not in result or c['after_meeting_id'] not in result:return False
            first=slots[result[c['before_meeting_id']]['slot']];last=slots[result[c['after_meeting_id']]['slot']]
            if first['day'] not in days or last['day'] not in days:raise TaskEvidenceError('unsupported calendar day')
            if (days[first['day']],first['end'])>(days[last['day']],last['start']):return False
        return True


class OrderingReceiptTask(ReceiptTask):
    def __init__(self,world,source_root):
        if world['kind']!='meeting':raise ValueError('ordering adapter is meeting-only')
        self.world=deepcopy(world);self.kernel=OrderingMeetingKernel('meeting',source_root,horizon=30)
        self.kernel.mailbox_acl['Worker']={'Worker'};self.expected=deepcopy(self.kernel.expected)
        self.binding={'version':self.version+'_ordering_source_extension','world':self.world,
            'source_hashes':self.kernel.source_hashes,'source_acl':{k:sorted(v) for k,v in self.kernel.source_acl.items()},
            'cost':0,'deadline':None,'source_native_obligation':False}
