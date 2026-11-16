"""Two additional native PiSAs operational constraints, preserving their semantics."""
from copy import deepcopy
from .ordering_receipt_tasks import OrderingMeetingKernel,OrderingReceiptTask
from .dynamic_tasks import TaskEvidenceError

DAYS={day:i for i,day in enumerate(('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))}
EXTRA={'meeting_requires_room_feature':('person','type','meeting_id','room_feature'),
       'must_finish_before_deadline':('person','type','meeting_id','deadline_day','deadline_hour')}

class ExtendedMeetingKernel(OrderingMeetingKernel):
    def business_constraints(self):
        fields={**EXTRA,'cannot_use_room':('person','type','room_id'),'person_unavailable':('person','type','slot_id'),
                'meeting_ordering':('person','type','before_meeting_id','after_meeting_id')}
        rows=[]
        for c in self.utility['hidden_private_constraints']:
            if c['type'] not in fields:raise TaskEvidenceError('unsupported source constraint type')
            rows.append({k:c[k] for k in fields[c['type']]})
        return rows

    @staticmethod
    def validate_constraints(rows):
        if not isinstance(rows,list):raise TaskEvidenceError('typed constraints required')
        for row in rows:
            if not isinstance(row,dict):raise TaskEvidenceError('typed constraint object required')
            kind=row.get('type')
            if kind not in EXTRA:OrderingMeetingKernel.validate_constraints([row]);continue
            if set(row)!=set(EXTRA[kind]):raise TaskEvidenceError('invalid extended constraint fields')
            if any(not isinstance(v,str) or not v for k,v in row.items() if k!='deadline_hour'):raise TaskEvidenceError('invalid extended string fields')
            if kind=='must_finish_before_deadline' and (row['deadline_day'] not in DAYS or type(row['deadline_hour']) not in (int,float) or not 0<=row['deadline_hour']<=24):raise TaskEvidenceError('invalid deadline')

    @staticmethod
    def schedule_valid(result,constraints,public):
        basic=[c for c in constraints if c['type'] not in EXTRA]
        if not OrderingMeetingKernel.schedule_valid(result,basic,public):return False
        rooms={r['id']:r for r in public['rooms']};slots={s['id']:s for s in public['slots']}
        for c in constraints:
            if c['type'] not in EXTRA:continue
            if c['meeting_id'] not in result:return False
            assignment=result[c['meeting_id']]
            if c['type']=='meeting_requires_room_feature':
                if c['room_feature'] not in rooms[assignment['room']]['features']:return False
            else:
                slot=slots[assignment['slot']]
                if slot['day'] not in DAYS:raise TaskEvidenceError('unsupported calendar day')
                if (DAYS[slot['day']],slot['end'])>(DAYS[c['deadline_day']],c['deadline_hour']):return False
        return True

class ExtendedMeetingReceiptTask(OrderingReceiptTask):
    def __init__(self,world,source_root):
        if world['kind']!='meeting':raise ValueError('meeting only')
        self.world=deepcopy(world);self.kernel=ExtendedMeetingKernel('meeting',source_root,horizon=30)
        self.kernel.mailbox_acl['Worker']={'Worker'};self.expected=deepcopy(self.kernel.expected)
        self.binding={'version':self.version+'_extended_native_constraints','world':self.world,'source_hashes':self.kernel.source_hashes,
            'source_acl':{k:sorted(v) for k,v in self.kernel.source_acl.items()},'cost':0,'deadline':None,'source_native_obligation':False}
