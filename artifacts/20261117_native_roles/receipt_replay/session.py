"""Role-local transport over pinned native telecom tools; no evaluator data in views."""
import copy,json,os,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];SOURCE=R.parent/'20261117_action_binding/tau2_source'
os.environ['PYTHON_DOTENV_DISABLED']='1'
sys.path[:0]=[str(ROOT/'.cache/tau2_deps_20261117'),str(SOURCE/'src')]
from tau2.domains.telecom.environment import get_environment

def plain(value):
    return json.loads(json.dumps(value,default=lambda x:x.model_dump(mode='json') if hasattr(x,'model_dump') else str(x)))

class RoleSession:
    def __init__(self,task):
        self.env=get_environment();initial=task.initial_state
        if initial:
            if initial.message_history:raise ValueError('prepopulated conversation not supported by this pinned-subset adapter')
            self.env.set_state(initial.initialization_data,initial.initialization_actions,[])
        self.instructions={'assistant':self.env.get_policy(),'user':str(task.user_scenario)}
        self.schemas={'assistant':[t.openai_schema for t in self.env.get_tools()],'user':[t.openai_schema for t in self.env.get_user_tools()]}
        self.local={'assistant':[],'user':[]};self.events=[]

    def view(self,role):
        if role not in self.local:raise ValueError('invalid role')
        return copy.deepcopy({'role':role,'instructions':self.instructions[role],'tools':self.schemas[role],'history':self.local[role]})

    def tool(self,role,name,arguments):
        if role not in self.local:raise ValueError('invalid role')
        before=[self.env.get_db_hash(),self.env.get_user_db_hash()]
        try:
            result=plain(self.env.make_tool_call(name,requestor=role,**arguments));self.env.sync_tools();error=None
        except Exception as exc:result=None;error=type(exc).__name__+': '+str(exc)
        after=[self.env.get_db_hash(),self.env.get_user_db_hash()];entry={'type':'tool','name':name,'arguments':copy.deepcopy(arguments),'result':result,'error':error};self.local[role].append(entry);self.events.append({'type':'tool','role':role,'name':name,'before':before,'after':after,'database_changed':before!=after,'error':error});return copy.deepcopy(entry)

    def message(self,role,text,deliver=True):
        if role not in self.local or not isinstance(text,str) or not text.strip():raise ValueError('invalid message')
        other='user' if role=='assistant' else 'assistant';entry={'type':'message','speaker':role,'content':text};self.local[role].append(copy.deepcopy(entry))
        if deliver:self.local[other].append(copy.deepcopy(entry))
        self.events.append({'type':'message','role':role,'delivered':deliver})

    def counters(self):
        return {'submitted_tool_calls':sum(e['type']=='tool' for e in self.events),'mutating_tool_calls':sum(e.get('database_changed',False) for e in self.events),'messages_sent':sum(e['type']=='message' for e in self.events),'messages_delivered':sum(e['type']=='message' and e['delivered'] for e in self.events)}
