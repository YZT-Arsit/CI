"""Execute student choices in existing PiSAs, relational, scope and temporal environments."""
import dataclasses,random
from fractions import Fraction
from functools import lru_cache
from prepare_multi import *
sys.path.insert(0,str(CORE/"tools"))
import audit_existing_602 as aud
sys.path[:0]=[str(aud.SCOPE/"code"),str(aud.TEMP/"code")]
import scope_env,temporal_env
class Missing(Exception):pass
class NativeChance(Exception):pass
class StudentRel(aud.ExactRel):
    def _random_uniform(self,event_type,attempt=0):
        if (event_type,attempt) not in self.bits:raise NativeChance(event_type,attempt)
        return super()._random_uniform(event_type,attempt)
@lru_cache(None)
def load_data():
    inv={r["row_id"]:r for f in sorted((V9/"typed").glob("*.jsonl")) for r in rows(f)}
    pairs={r["pair_id"]:r for r in rows(aud.P/"dataset/manifests/PISAS_TRAIN_MULTIPAIR_PRE_REG_V1/PREREG_TRAIN_ALL_SOURCE_PAIRS.jsonl")}
    parents={r["root_id"]:r for r in rows(aud.P/"dataset/manifests/PISAS_PROSPECTIVE_COMPOSITION_POOL_V1/PREREG_TRAIN_ORIGINAL_SOURCE_CANDIDATES.jsonl")}
    specs={}
    for p in (aud.REL/"data/PREREG_RELATIONAL_TASKS_V2/PREREG_TRAIN_MECHANISMS_WITH_QUERIES.jsonl",aud.SCOPE/"data/PREREG_RECIPIENT_SCOPE_V1/TRAIN_SPECS.jsonl",aud.TEMP/"data/PREREG_TEMPORAL_FULL_BUSINESS_V1/TRAIN_SPECS.jsonl"):
        specs.update((r["mechanism_id"],r) for r in rows(p))
    return inv,pairs,parents,specs,rd(aud.P/"dataset/manifests/PISAS_OFFICIAL_FULL_DOWNLOAD.json")
@lru_cache(None)
def original(p):
    return rows(Path(p))
def choose_cases():
    labs=rgz(OUT/"supervision.gz");used=set();result=[]
    for l in sorted(labs,key=lambda r:key(["env-v1",r["row"]])):
        k=l["mechanism"]
        family=("pisas" if k=="PISAS_LITERAL_SOURCE_CONJUNCTION" else "relational" if k.startswith(("REL_DIRECT","REL_TWO_HOP","REL_RARE")) else "temporal" if k=="PISAS_TEMPORAL_FULL_BUSINESS_QMU" else "scope" if k=="PISAS_SCOPE_SAFE_DELIVERY_QMU" else None)
        if not family or l["partition"]!="dev" or (family,l["root"]) in used:continue
        used.add((family,l["root"]));result.append((family,l))
    return result
@lru_cache(None)
def instance(family,row,auth=False):
    inv,pairs,parents,specs,pin=load_data();rec=inv[row]
    old=original(rec["source_file"])[rec["source_line_zero_based"]]
    assert old["split"]=="train"
    if family=="pisas":
        s=pairs[old["pair_id"]];w=aud.load_world(aud.P/"dataset/sources/pisas",parents[s["root_id"]],pin)
        return aud.TaskSufficientCompositionEnv(w,s["first_fact"],s["second_fact"],first_proof_cost=s["proof_costs"][0],second_proof_cost=s["proof_costs"][1],relation_authorized=auth),s
    s=specs[old["mechanism_id"]]
    if family=="relational":
        src=aud.R/"kci_public_refs_20261119/MultiAgentPrivacy/data"/s["source_file"];assert sha(src)==s["source_sha256"];return s,rd(src)
    w=aud.load_world(aud.P/"dataset/sources/pisas",s["source_record"],pin)
    return (scope_env.ScopeEnv(w,s) if family=="scope" else temporal_env.Env(w,s)),s
def ask(pred,pending,ctx,aa):
    acts=sorted({cn(a) for a in aa});assert acts
    t=key([ctx,acts])
    if t not in pred:
        pending[t]=dict(ctx=ctx,actions=acts);raise Missing(t)
    z=pred[t];assert len(z)==len(acts)
    return json.loads(acts[max(range(len(z)),key=z.__getitem__)])
def dynamic(models):
    pending={};results=[];local=[];cases=choose_cases()
    for family,l in cases:
        row=l["row"];r=l["root"]
        env,spec=instance(family,row)
        for name,pred in models.items():
            def pick(ctx,aa):return ask(pred,pending,ctx,aa)
            def emit(sub,mass,success,risk,abort,cost,tr,unknown=False):
                results.append(dict(model=name,family=family,root=r,row=row,condition=sub,mass=float(mass),stsr=int(success and not risk and not unknown),risk=int(risk),abort=int(abort),cost=cost,unknown=int(unknown),trace=tr))
            if family=="pisas":
                for auth in (False,True):
                    e,s=instance(family,row,auth);ref=aud.TaskSufficientPublicContinuation();st,prefix=aud.to_source(e,ref,s["first_fact"])
                    aa=e.legal_actions(st)
                    for kind in (aud.RAW,aud.SIGNAL_KIND):
                        a=next(a for a in aa if a["kind"]==kind);n,delivery=e.transition(st,a);b,_=aud.to_source(e,ref,s["second_fact"],at=n)
                        opts=e.legal_actions(b);rw=next(a for a in opts if a["kind"]==aud.RAW)
                        try:act=pick(e.observation(b),opts)
                        except Missing:continue
                        aft,rep=e.transition(b,act);v=e.evaluate_CI(b,rw)["verdict"]
                        local.append(dict(model=name,family=family,root=r,auth=auth,history=kind,raw_verdict=v,raw_selected=act==rw,actual_action=act,verdict=rep["verdict"],sent=bool(rep.get("real_source_delivery_event",{})),first_delivery=delivery))
                st=env.initial_state();trace=[];unknown=False
                try:
                    for step in range(st.max_turns+2):
                        if st.done:break
                        ob=env.observation(st);a=pick(ob,env.legal_actions(st));after,rep=env.transition(st,a)
                        trace.append(dict(state=env.state_id(st),action=a,verdict=rep["verdict"],delivery=rep.get("real_source_delivery_event"),after=env.state_id(after)))
                        st=after
                        if rep["verdict"] in ("UNKNOWN","INVALID"):unknown=True;break
                    emit("native",1,st.success,st.any_violation,st.aborted,st.cost,trace,unknown or not st.done)
                except Missing:pass
            elif family=="relational":
                s,src=env,spec
                chance=[({},Fraction(1))]
                probs={"PROTECTED_PRIVATE_UNAVAILABLE":Fraction(str(s["policy_regime"]["private_protected_channel_unavailable_probability"])),"CONSENT_GRANTED":Fraction(str(s["policy_regime"]["consent_grant_probability"]))}
                while chance:
                    bits,mass=chance.pop()
                    e=StudentRel(s,src,bits);st=e.initial_state();trace=[]
                    try:
                        for _ in range(24):
                            if st.done:break
                            ob=e.observation(st);a=pick(ob,e.legal_actions(st));n,v=e.transition(st,a)
                            trace.append(dict(action=a,state=e.state_id(st),verdict=v["verdict"],events=n.event_log[len(st.event_log):]));st=n
                        emit(str(bits),mass,st.success,st.any_violation,st.aborted,st.cost,trace,not st.done)
                    except NativeChance as event:
                        prob=probs[event.args[0]]
                        for yes,weight in ((True,prob),(False,1-prob)):
                            if weight:chance.append(({**bits,event.args:yes},mass*weight))
                    except Missing:pass
            else:
                # Four predeclared common pseudorandom environment draws from the native chance mass.
                chances=[(av,w,den) for av,w,den in env.chance() if w>0]
                gen=random.Random(key(["native-chance",r,family]));draws=gen.choices(chances,weights=[b for a,b,c in chances],k=4)
                for j,(av,_,den) in enumerate(draws):
                    try:
                        if family=="scope":
                            st=env.initial();a=pick(env.observation(st),env.actions(st))
                            o=scope_env.run(env,a,av,lambda e,st:pick(e.observation(st),e.actions(st)))
                            emit(str(j),.25,o["task_success"],o["CI_violation"],o["aborted"],o["steps"],o["trace"])
                        else:
                            st=env.initial();a=pick(env.observation(st),env.actions(st));st=env.after_first(a,av);trace=[dict(first=a,revealed=av,receipts=st.receipts)]
                            for _ in range(64):
                                if st.done:break
                                ob=env.observation(st);a=pick(ob,env.actions(st));n=env.step(st,a);trace.append(dict(action=a,new_receipts=n.receipts[len(st.receipts):],events=n.events[len(st.events):]));st=n
                            emit(str(j),.25,st.full_success,st.violation,st.aborted,st.steps,trace,not st.done)
                    except Missing:pass
    return pending,results,local
if __name__=="__main__":
    p,r,l=dynamic({"unscored":{}});print("bound",Counter(f for f,x in choose_cases()),len(p))

