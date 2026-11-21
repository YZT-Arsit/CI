r"""Shared CPU-TRAINABLE, ROLE-CONDITIONED actor on actual A->R->B->C simulator.

Single param set scores critical actions of A/RecordsAgent, B/Coordinator,
source agent (raw vs private), C/BusinessExecutor. Non-decision tool steps
(INSPECT_BUSINESS, correct business route, checking required source, signed
certificate) are executed by a fixed native adapter for ALL methods; this
is NOT an unrestricted multi-agent LLM execution.

Actor information is obtained exclusively from env.observation() and actions
from env.options(). Explicit differentiable Horn-rule inference is built into
the architecture and is a STRONG inductive bias, not learned open-domain CI.
"""
from __future__ import annotations
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import json,re,sys,numpy as np,torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import BASE_FACTS,SLOT_LABELS,FINAL,digest,task_catalog
from kci.upstream_downstream_chain_pilot import ChainPilotEnv
from kci.diverse_ci_long_horizon_v31 import enumerate_specs_for_root
from solve_v31_longterm import native_best
torch.set_num_threads(2)
NODE_NAMES=[f"S{i:02d}" for i in range(1,7)]+["X","Y","Z","protected association"]
NODE={key:i for i,key in enumerate(NODE_NAMES)}
MAX_RULES=12
ROLES=("RecordsAgent","Coordinator","DirectoryAgent","OrderAgent",
       "ComplianceAgent","BusinessExecutor","AuditBroker","Other")
STAGES=("A_SEND","preflight","release","choose_fact","other")
TYPES=("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B","OPEN_PRIVATE_CHANNEL",
       "SKIP_PRIVATE_CHANNEL","SEND_RAW_R","VERIFY_PRIVATE_R",
       "COMMIT_RESULT","ABORT","OTHER")
ACTIONS={name:i for i,name in enumerate(TYPES)}
TASKS={t.id:t for t in task_catalog()}
TEMPLATES={
    "calendar":"early_open","retail":"pending_yes","document":"external",
}

def spec_from_world(root,world):
    variant=int(world.split("_M")[-1].split("__")[0])-1
    task=TASKS[root]
    return enumerate_specs_for_root(task,TEMPLATES[task.business],variant)

def environment(row):
    return ChainPilotEnv(
        spec_from_world(row["root"],row["world"]),
        swap=row["swapped_package_names"],
        source_mode=row["source_mode"],
        safe_surcharge=row["source_cost_surcharge"])

def actor_feature(obs,action):
    """No access to hidden native state, oracle labels, saved Q or world classes."""
    act=str(action)
    is_A="source_bundle_tool_output" in obs
    history=obs.get("delivered_message_log",[])
    owner=obs.get("current_case_subject","")
    known=np.zeros(6,np.float32)
    for item in history:
        m=re.fullmatch(
            r"Certified archived task evidence for case (CASE-[A-F0-9]{8}): "
            r"source (S0[1-6]) was delivered to reviewer R\.",item["text"])
        if m is not None:
            if m.group(1)==owner:known[int(m.group(2)[1:])-1]=1.
            continue
        # Actual downstream native source-tool delivery is a different,
        # still certified text format, and is bound to the current task
        # through the role's signed source record, not to archived history.
        raw=re.search(r"Certified (?:calendar source|order record|project archive) "
                      r"DG[0-9]{2}: signal (S0[1-6]) ",item["text"])
        if raw:
            known[int(raw.group(1)[1:])-1]=1.
            continue
        if item["text"].startswith((
            "Signed private check confirms",
            "Trusted broker verified")):
            # Attestation is NOT a raw fact; never inject it into K(R).
            continue
        raise ValueError("Unparseable delivered source or certificate in actor observation")
    clauses=obs.get("public_inference_rules",obs.get("declared_knowledge_rules",[]))
    prem=np.zeros((MAX_RULES,len(NODE)),np.float32)
    head=np.zeros((MAX_RULES,len(NODE)),np.float32)
    assert len(clauses)<=MAX_RULES
    for i,rule in enumerate(clauses):
        for fact in rule["requires"]:prem[i,NODE[fact]]=1.
        head[i,NODE[rule["entails"]]]=1.
    required=obs.get("downstream_required_business_checks")
    if required is None:required=obs.get("task_evidence_contract",{}).get("required_slots",[])
    req=np.zeros(6,np.float32)
    for f in required:req[int(f[1:])-1]=1.
    src=np.zeros(6,np.float32)
    doc=obs.get("most_recent_tool_output") or {}
    if doc and isinstance(doc,dict):
        mt=re.search(r"\bsignal (S0[1-6])\b",doc.get("source_document",""))
        if mt:src[int(mt.group(1)[1:])-1]=1.
    bundle=np.zeros(6,np.float32)
    bundle_cost=0.
    if act in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"):
        packages=obs.get("source_bundle_tool_output") or {}
        name="PACKAGE_A" if act.endswith("A") else "PACKAGE_B"
        pkt=packages.get(name,{})
        for f in pkt.get("fact_slots",[]):
            bundle[int(f[1:])-1]=1.
        bundle_cost=float(pkt.get("cost_steps",0))
    role=obs.get("actor",obs.get("role","Other"))
    roleone=np.zeros(len(ROLES),np.float32)
    roleone[ROLES.index(role if role in ROLES else "Other")]=1
    stage=obs.get("stage",obs.get("current_stage","other"))
    stageone=np.zeros(len(STAGES),np.float32)
    stageone[STAGES.index(stage if stage in STAGES else "other")]=1
    actionone=np.zeros(len(TYPES),np.float32)
    actionone[ACTIONS.get(act,ACTIONS["OTHER"])]=1
    c=obs.get("privacy_resource_contract",{})
    auth=obs.get("privacy_policy",obs.get("recipient_policy",""))
    cost=float(c.get("step_cost",0.03))
    setup=float(c.get("private_channel_setup_steps",3))
    avail=obs.get("remaining_steps",obs.get("task_budget_remaining",15))
    bus=obs.get("public_business_state",{})
    extra=np.array([
        min(len(history)/5,1),len(required)/6,setup/5,cost/0.05,
        float(avail)/25,bundle_cost/6,
        float("Authorization exception is explicitly granted" in auth),
        float(bus.get("route")=="early"),float(bus.get("route")=="late"),
        float(bus.get("route")=="cancel"),float(bus.get("route")=="return"),
        float(bus.get("route")=="direct"),
        float(obs.get("private_channel_open",False)),
        float(action=="SEND_RAW_R"),float(action=="VERIFY_PRIVATE_R"),
        float(action=="ABORT"),float(action=="COMMIT_RESULT"),
        float(is_A),
    ],np.float32)
    return {"known":known,"req":req,"src":src,"bundle":bundle,
            "prem":prem,"head":head,"role":roleone,"stage":stageone,
            "action":actionone,"extra":extra}

def stack(feat):
    return {k:torch.tensor(np.stack([x[k] for x in feat]),dtype=torch.float32)
            for k in feat[0]}

class SharedHornActor(nn.Module):
    """One shared learned action scorer for all agents; the Horn propagation
    architecture is an explicit symbolic-neural prior, NOT learned language.
    """
    def __init__(self):
        super().__init__()
        self.rule_scale=nn.Parameter(torch.tensor(1.8))
        self.gain=nn.Sequential(
            nn.Linear(6*4+len(ROLES)+len(STAGES)+len(TYPES)+18+3,48),
            nn.Tanh(),nn.Linear(48,24),nn.Tanh(),nn.Linear(24,1))
        # The original network received the same future-risk scalar for
        # both OPEN and SKIP. It had to discover a conditional interaction
        # with each action from very few examples. Provide per-action risk
        # basis functions computed solely from public observations, while
        # learning their weights independently from each training method.
        self.risk_action_head=nn.Linear(5,1,bias=False)
        nn.init.zeros_(self.risk_action_head.weight)
    def infer(self,base,prem,head):
        known=torch.zeros((base.shape[0],len(NODE)),dtype=base.dtype,device=base.device)
        known[:,:6]=base.clamp(0,1)
        # The V3.1 baseline used sigmoid(k*(matched-premise_count+0.5)).
        # This fires *even when a required premise is ABSENT* (e.g. 0.2
        # activation for one missing premise), and five recursions amplify
        # those false edges until both safe and unsafe worlds look risky.
        #
        # Zero-preserving differentiable Horn semantics: a rule may fire
        # only if ALL of its declared antecedents are available. With
        # the native 0/1 source certificates, it is EXACT Boolean AND/OR
        # without consulting future states, labels, hidden closure or Qstar.
        #
        # The learnable exponent affects uncertain fractional inputs but
        # cannot fabricate facts from a missing premise. This is a very
        # strong public-rule inductive bias, NOT learned language inference.
        active=(head.sum(dim=-1)>0).to(dtype=known.dtype)
        exponent=F.softplus(self.rule_scale)+1e-6
        for _ in range(5):
            antecedents=(1-prem)+prem*known.unsqueeze(1)
            eligible=torch.prod(antecedents,dim=-1)*active
            fire=eligible.pow(exponent)
            inferred=1-torch.prod(1-fire.unsqueeze(-1)*head,dim=1)
            known=known+(1-known)*inferred
        return known[:,NODE["protected association"]]
    def forward(self,x):
        now=torch.maximum(x["known"],x["src"])
        fut=torch.maximum(x["known"],x["req"])
        futureA=torch.maximum(fut,x["bundle"])
        rn=self.infer(now,x["prem"],x["head"])
        rf=self.infer(fut,x["prem"],x["head"])
        ra=self.infer(futureA,x["prem"],x["head"])
        z=torch.cat((x["known"],x["req"],x["src"],x["bundle"],
                    x["role"],x["stage"],x["action"],x["extra"],
                    rn.unsqueeze(-1),rf.unsqueeze(-1),ra.unsqueeze(-1)),dim=-1)
        is_a=x["action"][:,ACTIONS["A_SEND_PACKAGE_A"]]+x["action"][:,ACTIONS["A_SEND_PACKAGE_B"]]
        interaction=torch.stack((
            ra*is_a,
            rf*x["action"][:,ACTIONS["OPEN_PRIVATE_CHANNEL"]],
            rf*x["action"][:,ACTIONS["SKIP_PRIVATE_CHANNEL"]],
            rn*x["action"][:,ACTIONS["SEND_RAW_R"]],
            rn*x["action"][:,ACTIONS["VERIFY_PRIVATE_R"]],
        ),dim=-1)
        return self.gain(z).squeeze(-1)+self.risk_action_head(interaction).squeeze(-1)

def valid_decision_candidates(obs,available):
    if obs.get("stage")=="A_SEND":
        return [x for x in available if x in ("A_SEND_PACKAGE_A","A_SEND_PACKAGE_B","ABORT_CHAIN")]
    stage=obs.get("current_stage")
    if stage=="preflight":
        return [x for x in available if x in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL","ABORT")]
    if stage=="release":
        return [x for x in available if x in ("SEND_RAW_R","VERIFY_PRIVATE_R","ABORT")]
    if stage=="choose_fact" and "COMMIT_RESULT" in available:
        return [x for x in available if x in ("COMMIT_RESULT","ABORT")]
    return []

def baseline_automatic_action(env,state):
    avail=list(env.options(state))
    if not avail:return None
    if "A_READ_VERIFIED_PACKAGES" in avail:return "A_READ_VERIFIED_PACKAGES"
    if "INSPECT_BUSINESS" in avail:return "INSPECT_BUSINESS"
    route="CHOOSE:"+env.native._route
    if route in avail:return route
    if "SEND_CERT_R" in avail:return "SEND_CERT_R"
    reads=[x for x in avail if x.startswith("READ:")]
    if reads:return sorted(reads)[0]
    return None

def score_actions(model,obs,options):
    feat=stack([actor_feature(obs,a) for a in options])
    scores=model(feat)
    return scores

def choose(model,obs,options,rng=None,greedy=True):
    with torch.no_grad():
        scores=score_actions(model,obs,options)
        probs=torch.softmax(scores,dim=0).detach().cpu().numpy()
    if greedy:idx=int(np.argmax(probs))
    else:
        assert rng is not None
        idx=int(rng.choice(len(options),p=probs/probs.sum()))
    return options[idx],float(np.log(max(float(probs[idx]),1e-12)))

def run_episode(env,model,random_rng=None,greedy=True,collect_logprob=False,
                max_native_steps=55):
    st=env.initial()
    events=[]
    likelihoods=[]
    decision_states=[]
    nchoices=0
    for _ in range(max_native_steps):
        if st.native.done:break
        obs=env.observation(st)
        avail=list(env.options(st))
        decision=valid_decision_candidates(obs,avail)
        if len(decision)>=2:
            # DO NOT substitute hidden truth for a policy's actual action.
            if collect_logprob:
                scores=score_actions(model,obs,decision)
                probs=torch.softmax(scores,dim=0)
                pp=probs.detach().numpy()
                idx=int(random_rng.choice(len(decision),p=pp/pp.sum()))
                act=decision[idx]
                likelihoods.append(torch.log(probs[idx]+1e-10))
            else:
                act,_=choose(model,obs,decision,random_rng,greedy)
            decision_states.append({"actor_observation":deepcopy(obs),
                "candidate_actions":list(decision),
                "selected_action":act,
                "native_state_before":deepcopy(st.native),
                "chain_state_before":deepcopy(st)})
            nchoices+=1
            actor=obs.get("actor",obs.get("role"))
            choice=True
        else:
            act=baseline_automatic_action(env,st)
            actor=obs.get("actor",obs.get("role"))
            choice=False
        if act is None:
            return {"success":False,"violation":st.native.violation,
                    "done":st.native.done,"invalid":True,
                    "return":0.0,"actions":events,"n_policy_choices":nchoices,
                    "logp":likelihoods,"decision_states":decision_states,
                    "final":st.native}
        before=digest(asdict(st.native))
        nxt,a=env.transition(st,act)
        events.append({"role":actor,"stage":obs.get("stage",obs.get("current_stage")),
                       "action":act,"learned_decision":choice,
                       "before_native_hash":before,"verdict":a["verdict"],
                       "after_native_hash":digest(asdict(nxt.native)),
                       "new_R_delivery":len(nxt.native.delivered["R"])-len(st.native.delivered["R"])})
        if a["verdict"] not in ("VALID","VIOLATION"):
            return {"success":False,"violation":st.native.violation,
                    "done":st.native.done,"invalid":True,"return":0.0,
                    "actions":events,"n_policy_choices":nchoices,
                    "logp":likelihoods,"decision_states":decision_states,
                    "final":st.native}
        st=nxt
    success=bool(st.native.done and st.native.success and not st.native.violation)
    violation=bool(st.native.violation)
    payoff=round(1-env.spec.cost_per_step*st.native.steps,6) if success else (
        -1.0 if violation else 0.0)
    return {"success":success,"violation":violation,"done":st.native.done,
            "invalid":False,"return":payoff,
            "native_steps":st.native.steps,
            "actions":events,"n_policy_choices":nchoices,
            "logp":likelihoods,"decision_states":decision_states,
            "final":st.native}

def gold_record(world):
    env=environment(world)
    return env
