"""Seal the complete 2^4 presentation factorial on the previous diagnostic worlds."""
import hashlib,itertools,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];PREV=R.parent/'20261117_action_binding';sys.path.insert(0,str(ROOT/'code'))
from kci.relation_language import prompt,seed_for,vocabulary
from experiments.relation_solution_data import environment
from render import render

def main():
    assert not (R/'PLAN.json').exists();source=PREV/'alias_confirmation/worlds.json';worlds=json.loads(source.read_text(encoding='utf-8'));palette=json.loads((R.parent/'20261116_relation_solution/entity_palettes.json').read_text())['fresh'];profiles=[dict(zip(['names','facts','plans','options'],bits)) for bits in itertools.product((0,1),repeat=4)];cases=[];equality=0;semantic=0
    for group,pair in enumerate(worlds):
        for side,w in enumerate(pair):
            env=environment(w);obs=env.observation(env.initial());nonces=[seed_for(w['base_group_id'],900+i) for i in range(2)]
            for view in range(2):
                seeds={k:nonces[view] for k in ['names','facts','links','plans','options']}
                for question,candidate in [('policy',None)]+[('task',a) for a in obs['actions']]:assert render(obs,palette,seeds,question,candidate)==prompt(obs,palette,nonces[view],question,candidate);equality+=1
            for profile in profiles:
                seeds={k:nonces[v] for k,v in profile.items()};seeds['links']=nonces[0];text,actions=render(obs,palette,seeds);inverse={v:k for k,v in vocabulary(obs,palette,seeds['names']).items()};fact_text=text.split('R received: ',1)[1].split('.\nPlans:',1)[0];facts=[]
                if fact_text!='nothing':
                    for fact in fact_text.split('; '):
                        tag,x,y=fact.split();facts.append((tag,inverse[x],inverse[y]))
                expected=[(tag,*edge) for message in obs['delivered'] if message['recipient']=='R' for field,tag in [('identities','I'),('links','L'),('histories','H')] for edge in message['content'].get(field,[])];assert sorted(facts)==sorted(expected)
                actual=[]
                for plan in text.split('Plans: ',1)[1].split('.\nOptions:',1)[0].split(' | '):
                    ticket,edges=plan.split(': ',1);links=[]
                    for edge in edges.split('; '):
                        tag,x,y=edge.split();assert tag=='L';links.append((inverse[x],inverse[y]))
                    actual.append((inverse[ticket],sorted(links)))
                assert sorted(actual)==sorted((p['ticket'],sorted(map(tuple,p['links']))) for p in obs['contract']['providers']);assert set(actions)==set(obs['actions']);semantic+=1
            cases.append({'group':group,'side':side,'world':w,'observation':obs,'nonces':nonces})
    data=R/'cases.json';data.write_text(json.dumps(cases),encoding='utf-8');plan={'scope':'same-DEV complete factorial attribution, not a new performance test; fixed one-hop coverage/class_balanced checkpoint; no training','factors':['names','facts','plans','options'],'profiles':profiles,'states':len(cases),'independent_world_pairs':16,'expected_query_outputs':1024,'expected_action_outputs':512,'links_note':'one edge per provider, so edge-order intervention is a no-op; fixed to base seed','case_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'cases_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'byte_equal_original_renders_checked':equality,'independent_prompt_graph_checks':semantic,'initial_checkpoint':'../20261117_action_binding/coverage_repair/class_balanced','renderer_sha256':hashlib.sha256((R/'render.py').read_bytes()).hexdigest(),'whole_goal_complete':False};(R/'PLAN.json').write_text(json.dumps(plan,indent=2));print(json.dumps({k:v for k,v in plan.items() if k!='profiles'}))

if __name__=='__main__':main()
