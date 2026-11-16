"""Frozen-checkpoint falsification of stage-3 relation/dependency claims.

Relabelled worlds are isomorphic diagnostics, not independent source tasks.
The transport wrapper preserves the old engine, including its dependency defects.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import random
import numpy as np
from kci.legacy_dynamic_tasks_v1 import DynamicTask, DynamicState, canonical
from experiments.text_policy_probe import text_features

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'artifacts/20261113_iteration2/source_anchors'
OLD=ROOT/'artifacts/20261113_stage3'


def rename(value,mapping):
    if isinstance(value,dict):return {k:rename(v,mapping) for k,v in value.items()}
    if isinstance(value,list):return [rename(v,mapping) for v in value]
    if isinstance(value,tuple):return tuple(rename(v,mapping) for v in value)
    return mapping.get(value,value) if type(value) in (str,int) else value


SWAP={'case8':'case9','case9':'case8',201:901,202:902,203:903,901:201,902:202,903:203}


def make_map(index,paired_stem=False):
    if index==0:return {}
    if index==1:return dict(SWAP)
    rng=random.Random(20261115+index)
    def word(n):return ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(n))
    result={'case8':word(4)+'1','case9':word(4)+'2','e7':word(1)+'3','eli':word(3)}
    if paired_stem:result['case9']=result['case8'][:-1]+'2'
    original=[201,202,203,901,902,903,701,702,703]
    fresh=rng.sample([v for v in range(100,1000) if v not in original],len(original))
    result.update(zip(original,fresh))
    for name in ('John Smith','Maria Lopez','James Chen'):
        result[name]=' '.join(word(len(part)) for part in name.split())
    assert len(set(result.values()))==len(result)
    assert all(len(str(k))==len(str(v)) for k,v in result.items())
    return result


def independent_join(kind,ledger):
    """Nested joins over delivered relations; independent of dictionary closure."""
    data=[m['content'] for m in ledger if m['recipient']=='R'];result=set()
    if kind=='meeting':
        for a in data:
          for b in data:
            for c in data:
              if (a['type']=='roster' and b['type']=='case_link' and c['type']=='case_cause'
                  and a['alias']==b['alias'] and b['case']==c['case']):result.add((a['person'],c['cause']))
    else:
        names=[r for m in data if m['type']=='owner_names' for r in m['rows']]
        links=[r for m in data if m['type']=='owner_index' for r in m['rows']]
        addresses=[r for m in data if m['type']=='city_bundle' for r in m['addresses']]
        for a in names:
          for b in links:
            for c in addresses:
              if a['company_id']==b['company_id'] and b['owner_id']==c['owner_id']:
                result.add((a['owner_name'],c['address']))
    return result


class BindingProbe:
    def __init__(self,kind,related,horizon,mapping=None,route_flip=False,allow=False):
        self.base=DynamicTask(kind,SOURCE,related=related,allow=True,horizon=horizon)
        self.kind=kind;self.mapping=mapping or {};self.inverse={v:k for k,v in self.mapping.items()}
        self.route_flip=route_flip;self.allow=allow

    def options(self,s):return self.base.options(s)

    def transition(self,s,action):
        before=independent_join(self.kind,s.ledger)
        decoded=DynamicState(**rename(asdict(s),self.inverse))
        n,audit=self.base.transition(decoded,action)
        if audit['verdict']=='UNKNOWN':return deepcopy(s),audit
        if self.route_flip:
            for m in n.ledger[len(decoded.ledger):]:
                if m['content']['type'] in ('case_link','owner_index'):
                    m['content']=rename(m['content'],SWAP)
        n=DynamicState(**rename(asdict(n),self.mapping))
        closure=self.base.protected(n.ledger)
        assert closure==independent_join(self.kind,n.ledger)
        new=closure-before
        if new and not self.allow:n.violation=True;n.done=True;n.success=False
        return n,{'verdict':'VIOLATION' if new and not self.allow else 'VALID',
                  'new_protected_relations':sorted(new)}

    def decision_start(self):return self.transition(DynamicState(),'DELIVER_HISTORY')[0]

    def observation(self,s):
        obs=self.base.observation(DynamicState(**rename(asdict(s),self.inverse)))
        if self.route_flip:
            obs['authorized_routing_metadata']=rename(obs['authorized_routing_metadata'],SWAP)
        obs['derived_relation_permission']='allowed' if self.allow else 'forbidden at R'
        return rename(obs,self.mapping)

    def exact(self,s):
        queue=[(deepcopy(s),[])];successes=[]
        while queue:
            state,path=queue.pop()
            if state.done:
                if state.success and not state.violation:successes.append((state.turn,path))
                continue
            for action in self.options(state):
                n,audit=self.transition(state,action)
                if audit['verdict']=='VALID':queue.append((n,path+[action]))
        return {'cost':min((v[0] for v in successes),default=None),
                'witness':min(successes,default=(None,None))[1]}


def generate(index,output,paired_stem=False):
    mapping=make_map(index,paired_stem);rows=[]
    for kind in ('meeting','query'):
      for horizon in ((6,9) if kind=='meeting' else (7,10)):
       for flip in (False,True):
        for related in (False,True):
          env=BindingProbe(kind,related,horizon,mapping,flip);s=env.decision_start()
          values={};traces={}
          for action in env.options(s):
            n,audit=env.transition(s,action);assert audit['verdict']=='VALID'
            values[action]=env.exact(n)
            trace=[{'action':action,'state':asdict(n),'audit':audit}]
            for a in values[action]['witness'] or []:
                n,audit=env.transition(n,a);trace.append({'action':a,'state':asdict(n),'audit':audit})
            traces[action]=trace
          feasible=[a for a,v in values.items() if v['cost'] is not None]
          chosen=min(feasible,key=lambda a:values[a]['cost'])
          other=next(a for a in values if a!=chosen)
          assert values[other]['cost'] is None or values[chosen]['cost']<values[other]['cost']
          rows.append({'kind':kind,'horizon':horizon,'route_flip':flip,'history_related_original':related,
             'relation_matches':related!=flip,'prompt':canonical(env.observation(s)),
             'chosen':chosen,'rejected':other,'values':values,'traces':traces})
    Path(output).write_text(json.dumps({'map_index':index,'paired_stem':paired_stem,'map':list(mapping.items()),'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps({'map_index':index,'rows':len(rows)}))


def evaluate(data,model,output):
    path=OLD/'finite_v2'/model/'weights.npz';before=sha256(path.read_bytes()).hexdigest()
    z=np.load(path);w=z['weights'];actions=list(z['actions']);rows=[]
    for p in sorted(Path(data).glob('map_*.json')):
        doc=json.loads(p.read_text(encoding='utf-8'))
        for r in doc['rows']:
            scores=w@text_features(r['prompt']);c=actions.index(r['chosen']);b=actions.index(r['rejected'])
            rows.append({k:r[k] for k in ('kind','horizon','route_flip','history_related_original','relation_matches','chosen','rejected')}
                        |{'map_index':doc['map_index'],'margin':float(scores[c]-scores[b]),
                          'correct':bool(scores[c]>scores[b]),'prediction':r['chosen'] if scores[c]>scores[b] else r['rejected']})
    assert before==sha256(path.read_bytes()).hexdigest()
    Path(output).write_text(json.dumps({'model':model,'weights_sha256':before,'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps({'model':model,'evaluated':len(rows),'correct':sum(r['correct'] for r in rows)}))


def dependency_probe(output):
    # Each named record is the prerequisite of the NEXT consumer, not a random log.
    specs=[
      ('meeting','private_request','PRIVATE_BOOKING',('ACK_PRIVATE','SEND_CASE_LINK','RESERVE','DELIVER_RESULT'),'private_booking'),
      ('meeting','sanitized_constraints','SEND_ROSTER REQUEST_SANITIZED COMPUTE_CONSTRAINTS',('SEND_CONSTRAINTS','RESERVE','DELIVER_RESULT'),'constraints'),
      ('meeting','reservation_receipt','PRIVATE_BOOKING ACK_PRIVATE SEND_CASE_LINK RESERVE',('DELIVER_RESULT',),'reservation'),
      ('query','projection_request','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION',('PROJECT_CITIES','SEND_CITIES','JOIN','DELIVER_RESULT'),'projection_request'),
      ('query','projection_ready','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES',('SEND_CITIES','JOIN','DELIVER_RESULT'),'projection_ready'),
      ('query','count_shard','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES',('JOIN','DELIVER_RESULT'),'counts'),
      ('query','city_shard','OPEN_INDEX FETCH_COUNTS REQUEST_CITY_PROJECTION PROJECT_CITIES SEND_CITIES',('JOIN','DELIVER_RESULT'),'cities'),
      ('query','vault_result_receipt','OPEN_VAULT FETCH_COUNTS FETCH_CITY_BUNDLE JOIN SEND_JOIN_RESULT',('DELIVER_RESULT',),'joined_result'),
    ]
    results=[]
    for kind,name,prefix,suffix,record_type in specs:
      env=DynamicTask(kind,SOURCE,related=True,horizon=20);initial=env.decision_start()
      for a in prefix.split():initial,audit=env.transition(initial,a);assert audit['verdict']=='VALID'
      for intervention in ('identity','delete','redirect','corrupt'):
        s=deepcopy(initial);matches=[m for m in s.ledger if m['content']['type']==record_type]
        assert matches
        if intervention=='delete':s.ledger=[m for m in s.ledger if m['content']['type']!=record_type]
        elif intervention=='redirect':
            for m in matches:m['recipient']='UnrelatedSink'
        elif intervention=='corrupt':
            for m in matches:
                if record_type=='counts':
                    for row in m['content']['rows']:row['gas_stations_count']+=100
                elif record_type=='cities':
                    for row in m['content']['rows']:row['city']='INVALID_CITY'
                elif 'schedule' in m['content']:m['content']['schedule']={}
                elif 'rows' in m['content']:m['content']['rows']=[]
                else:m['content']={'type':'INVALID_PAYLOAD'}
        start=asdict(s);trace=[];error=None
        try:
            for a in suffix:
                if s.done:break
                s,audit=env.transition(s,a);trace.append({'action':a,'audit':audit})
        except Exception as e:error=type(e).__name__+': '+str(e)
        results.append({'kind':kind,'input':name,'intervention':intervention,'before':start,
          'trace':trace,'after':asdict(s),'exception':error,
          'status':'EXCEPTION' if error else 'SAFE_SUCCESS' if s.success and not s.violation else 'REJECTED_OR_FAILED'})
    Path(output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'cases':len(results),'success_despite_cut':sum(r['intervention']!='identity' and r['status']=='SAFE_SUCCESS' for r in results)}))


def causal_probe(output):
    results=[]
    for kind in ('meeting','query'):
      for intervention in ('identity','delete_history','redirect_history','permission'):
        env=DynamicTask(kind,SOURCE,related=True,horizon=20,allow=intervention=='permission')
        s=env.decision_start()
        if intervention=='delete_history':s.ledger=[]
        if intervention=='redirect_history':s.ledger[0]['recipient']='UnrelatedSink'
        actions=('SEND_ROSTER','SEND_CASE_LINK') if kind=='meeting' else ('OPEN_INDEX','FETCH_CITY_BUNDLE')
        for a in actions:s,audit=env.transition(s,a)
        actual=env.protected(s.ledger);assert actual==independent_join(kind,s.ledger)
        results.append({'kind':kind,'intervention':intervention,'verdict':audit['verdict'],
                        'closure':sorted(actual),'ledger':s.ledger})
    Path(output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({'causal_cases':len(results)}))


def cost_probe(kind,output):
    rows=[]
    for related in (False,True):
        env=DynamicTask(kind,SOURCE,related=related,horizon=20);s=env.decision_start();paths={}
        for first in env.options(s):
            n,_=env.transition(s,first);stack=[(n,[first])];success=[]
            while stack:
                state,path=stack.pop()
                if state.done:
                    if state.success and not state.violation:success.append(path)
                    continue
                for a in env.options(state):
                    after,audit=env.transition(state,a)
                    if audit['verdict']=='VALID':stack.append((after,path+[a]))
            paths[first]=success
        for private_weight in range(5):
          for sanitize_weight in range(5):
            weights={a:private_weight for a in ('ACK_PRIVATE','SEND_JOIN_RESULT')}
            weights.update({a:sanitize_weight for a in ('REQUEST_SANITIZED','COMPUTE_CONSTRAINTS','REQUEST_CITY_PROJECTION','PROJECT_CITIES')})
            values={a:min((1+sum(weights.get(x,1) for x in p) for p in ps),default=None) for a,ps in paths.items()}
            best=min(x for x in values.values() if x is not None)
            rows.append({'kind':kind,'related':related,'private_overhead_weight':private_weight,
                         'sanitize_overhead_weight':sanitize_weight,'values':values,
                         'optimal_actions':[a for a,v in values.items() if v==best],
                         'both_routes_safely_feasible':all(paths.values())})
    Path(output).write_text(json.dumps({'rows':rows},indent=2),encoding='utf-8')
    print(json.dumps({'kind':kind,'cost_cases':len(rows)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode');p.add_argument('output')
    p.add_argument('--index',type=int);p.add_argument('--data');p.add_argument('--model');p.add_argument('--kind')
    p.add_argument('--paired-stem',action='store_true')
    a=p.parse_args()
    if a.mode=='generate':generate(a.index,a.output,a.paired_stem)
    elif a.mode=='evaluate':evaluate(a.data,a.model,a.output)
    elif a.mode=='dependency':dependency_probe(a.output)
    elif a.mode=='causal':causal_probe(a.output)
    elif a.mode=='cost':cost_probe(a.kind,a.output)
    else:raise ValueError(a.mode)
