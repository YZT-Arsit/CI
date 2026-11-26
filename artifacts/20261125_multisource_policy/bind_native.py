"""Bind existing source-specific legal menus to signed finite execution branches."""
from collections import defaultdict
from fractions import Fraction
from prepare_multi import *
sys.path.insert(0,str(CORE/"tools"))
import build_aggregate_core as ag
import build_negotiation_core as ng
sys.path.insert(0,str(ART.parent/"code"))
from kci.v9_execution import Execution
def tag(ctx,actions):
    acts=sorted(cn(a) for a in actions);k=key([ctx,acts]);visible[k]=dict(ctx=ctx,actions=acts);return k
def add_native(root,domain,t,pp,nn,mechanism):
    if not pp or not nn:return
    r=dict(tag=t,root=root,source=root,row="native-"+key([root,t,sorted(pp),sorted(nn)])[:24],domain=domain,tier="native_action_set",mechanism=mechanism,value_tier=None,partition="train",kind="masked",pos=sorted(pp),neg=sorted(nn),target=None)
    if r["row"] not in seen:extra.append(r);seen.add(r["row"])
class Negotiation(Execution):
    def ask(self,c,menu):return tag(c,menu)
def aggregate(r,s,permission):
    truth=ag.bits(s);ca=ag.context(s,truth,permission)
    acts=[ag.action(s,truth,0,k) for k in ("EXACT_COUNT","THRESHOLD_REPORT")]
    assert permission or cn(ca)==cn(r["actor_context"])
    head=tag(ca,acts);paths={}
    for a in acts:
        e,va=ag.audit(s,truth,[],a,permission)
        cb=dict(ca);cb["role"]="source_analyst_2";cb["local_authorized_view"]=[{"entity":s["public_entities"][i],"matches_protected_predicate":bool(truth[i])} for i in s["groups"][1]]
        cb.pop("recipient_actual_receipts",None)
        cb["observation_scope"]="Only the coordinator receives both reports. This analyst has its own cohort view and the public task."
        bb=[ag.action(s,truth,1,k) for k in ("EXACT_COUNT","THRESHOLD_REPORT")];stop={"kind":"SAFE_ABORT"}
        tb=tag(cb,bb+[stop]);out={}
        for b in bb+[stop]:
            abort=b==stop;evts=[e];risk=va["verdict"]=="VIOLATION";cost=s["costs"][a["kind"]][0]
            if not abort:
                ev,vb=ag.audit(s,truth,evts,b,permission);evts.append(ev);risk|=vb["verdict"]=="VIOLATION";cost+=s["costs"][b["kind"]][1]
            answer={ev["query_index"]:bool(ev["value"] if ev["kind"]=="THRESHOLD_REPORT" else ev["value"]>=s["thresholds"][ev["query_index"]]) for ev in evts}
            expected={i:sum(truth[j] for j in grp)>=s["thresholds"][i] for i,grp in enumerate(s["groups"])}
            success=answer==expected
            out[cn(b)]=dict(success=success,direct=False,inferred=risk,stsr=success and not risk,abort=abort,events=evts,cost=cost)
        paths[cn(a)]=[dict(condition="aggregate",probability="1",policy_input=tb,outcomes=out)]
    return dict(first_input=head,paths=paths)
def proof_assets(partition,gs,os,raws):
    mapping={}
    for t,ob in os.items():
        ctx=json.loads(ob["prompt"].split("\n",1)[1].split("\nExecutable actions:\n")[0])
        mapping[t]=tag(ctx,[json.loads(a) for a in ob["actions"]])
    for g in gs:
        if g["partition"]=="train" and g["root"] not in train_pw:continue
        for k,w in g["worlds"].items():
            if g["partition"]=="train":
                ta=mapping[w["head"]];add_native(g["root"],"PROOFWRITER",ta,w["accepted"],set(w["paths"])-set(w["accepted"]),k)
                for a in g["pair"]:
                    s=w["paths"][a];add_native(g["root"],"PROOFWRITER",mapping[s["tag"]],s["accepted"],set(s["classes"])-set(s["accepted"]),k)
        if g["partition"]!="train":raws[g["root"]]=g
    return mapping
def main():
    global visible,extra,seen,train_pw
    labs=rgz(OUT/"supervision.gz");visible={};extra=[];seen=set();train_pw={r["root"] for r in labs if r["domain"]=="PROOFWRITER" and r["partition"]=="train"}
    rows0={r["row_id"]:r for r in rows(CORE/"D_future_core_multisource_v2/D_future_core.jsonl")}
    selected=defaultdict(list)
    for r in labs:
        if r["row"] in rows0 and (r["domain"]=="DEAL_OR_NO_DEAL" or r["mechanism"]=="NESTED_COUNT_DIFFERENCING"):selected[r["partition"],r["domain"],r["root"]].append(r)
    chosen=[min(v,key=lambda r:key(["native-v1",r["row"]])) for v in selected.values()]
    nia={r["row"] for r in chosen if r["domain"]=="DEAL_OR_NO_DEAL"};na=Negotiation.__new__(Negotiation);na.ng=ng;na.neg={}
    with gzip.open(CORE/"data/NEGOTIATION_CORE_V2/EVAL_ONLY_ALL_NATIVE_BRANCHES.jsonl.gz","rt",encoding="utf8") as f:
        for line in f:
            v=json.loads(line)
            if v["id"] in nia:na.neg[v["id"]]=v
    ids={r["row"] for r in chosen if r["mechanism"]=="NESTED_COUNT_DIFFERENCING"}
    specs={r["spec_id"]:r for r in rows(CORE/"data/AGGREGATE_K_CORE_V1/PREREG_ALL_CANDIDATES.jsonl") if r["spec_id"] in ids}
    graphs=[]
    for lab in chosen:
        r=rows0[lab["row"]];dom=lab["domain"]
        if dom=="DEAL_OR_NO_DEAL":
            head=tag(r["actor_context"],r["executable_actions"])
            games={"original":dict(first_input=head,paths={cn(a):list(na.negotiation(r,a)) for a in r["executable_actions"]})}
        else:games={("authorized" if perm else "original"):aggregate(r,specs[r["row_id"]],perm) for perm in (False,True)}
        for kind,g in games.items():
            assert kind!="original" or g["first_input"]==lab["tag"]
            if lab["partition"]=="train" and dom=="DEAL_OR_NO_DEAL":
                # One chance event across both possible delivery histories; include the actual complete accepting alternatives.
                for a,bs in sorted(g["paths"].items()):
                    br=bs[int(key(["event",lab["root"]])[:8],16)%len(bs)];out=br["outcomes"]
                    pp=[b for b,v in out.items() if v["stsr"]] or [b for b,v in out.items() if v["abort"] and not(v["direct"] or v["inferred"])]
                    add_native(lab["root"],dom,br["policy_input"],pp,set(out)-set(pp),lab["mechanism"])
            if lab["partition"]=="train" and dom=="MULTIAGENTPRIVACY":
                # Indistinguishable B histories share one target, lawful and sufficient in every compatible branch.
                states=[cs[0] for cs in g["paths"].values()]
                assert len({b["policy_input"] for b in states})==1
                lawful=set.intersection(*[{a for a,v in b["outcomes"].items() if v["stsr"]} for b in states])
                add_native(lab["root"],dom,states[0]["policy_input"],lawful,set(states[0]["outcomes"])-lawful,lab["mechanism"])
        if lab["partition"]=="dev":graphs.append(dict(root=lab["root"],row=lab["row"],domain=dom,mechanism=lab["mechanism"],worlds=games))
    proofdev={};mp=proof_assets("existing",rgz(OLD/"graphs.gz"),rgz(OLD/"observations.gz"),proofdev)
    if (NEW/"fresh_graphs.gz").exists():mp.update(proof_assets("new",rgz(NEW/"fresh_graphs.gz"),rgz(NEW/"fresh_obs.gz"),proofdev))
    needed={r["tag"] for r in extra}
    for g in graphs:
        for w in g["worlds"].values():
            needed.add(w["first_input"]);needed.update(b["policy_input"] for branches in w["paths"].values() for b in branches)
    for g in proofdev.values():
        for w in g["worlds"].values():needed.add(mp[w["head"]]);needed.update(mp[b["tag"]] for b in w["paths"].values())
    train_tags={r["tag"] for r in labs+extra if r["partition"]=="train"}
    dev_tags={r["tag"] for r in labs if r["partition"]=="dev"}
    for g in graphs:
        for w in g["worlds"].values():
            dev_tags.add(w["first_input"]);dev_tags.update(b["policy_input"] for paths in w["paths"].values() for b in paths)
    for g in proofdev.values():
        for w in g["worlds"].values():
            dev_tags.add(mp[w["head"]]);dev_tags.update(mp[b["tag"]] for b in w["paths"].values())
    train_groups={r["root"] for r in labs+extra if r["partition"]=="train"}
    dev_groups={r["root"] for r in labs if r["partition"]=="dev"}|set(proofdev)
    assert not (train_tags&dev_tags or train_groups&dev_groups)
    dump(OUT/"NATIVE_BINDING.json",dict(train_tags=len(train_tags),development_tags=len(dev_tags),train_groups=len(train_groups),development_groups=len(dev_groups),common_groups=[],common_observations=[],aggregate_B_access="public task and own cohort only",negotiation_event_mass="seven equiprobable native chance outcomes"))
    gz(OUT/"native_inputs.gz",{t:visible[t] for t in needed});gz(OUT/"native_train.gz",extra)
    gz(OUT/"native_graphs.gz",graphs);gz(OUT/"proof_map.gz",mp);gz(OUT/"proof_dev.gz",proofdev)
    print("native",len(graphs),"additional training",len(extra),"inputs",len(needed),flush=True)
if __name__=="__main__":main()

