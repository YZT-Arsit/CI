r"""Exact CPU frozen-policy Qpi on the unchanged KCI A->R->B->C native simulator.

A source-choice fork: same verified A state, two immediately legal sends.
All later learned decisions use FROZEN checkpoint softmax over CURRENT legal
actions; the simulator's existing common mandatory-tool adapter is identical
to real run_episode. Enumerate all reachable branches, not Monte Carlo.
Policy never reads Qstar, case IDs, labels, hidden state, or future outcome.
Full role/phase/hash/native signed deliveries and database mutations are
independently verifiable in saved traces. No Qwen weights, no GPU.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from collections import Counter
from hashlib import sha256
import json,math,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from chain_shared_actor_cpu_action_interaction import (
 SharedHornActor,actor_feature,stack,environment,valid_decision_candidates,
 baseline_automatic_action,digest)
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from solve_v31_longterm import native_best
CKPT=ROOT/"artifacts"/"20261120_v31"/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"SMALL_CPU_CHECKPOINTS"
DATA=ROOT/"artifacts"/"20261120_v31"/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"
ROWS={r["world"]:r for s in ("train","dev","exploratory_test") for r in GROUP[s]}
SEEDS=(7,1234,2025)
KINDS=("action_sft","cce_reference_dpo_proxy")

def load_frozen_policy(kind,seed,*,path_override=None):
    assert kind in KINDS and seed in SEEDS
    file=path_override or CKPT/(kind+"_seed"+str(seed)+".pt")
    checkpoint=torch.load(file,map_location="cpu",weights_only=True)
    assert checkpoint["seed"]==seed and checkpoint["method"]==kind
    assert checkpoint["source_data_sha256"]==sha256(
      (DATA/"CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl").read_bytes()).hexdigest()
    model=SharedHornActor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model,{
      "kind":kind,"seed":seed,"checkpoint":file.name,
      "checkpoint_sha256":sha256(file.read_bytes()).hexdigest(),
      "source_data_sha256":checkpoint["source_data_sha256"],
      "trainable_model_params":sum(p.numel() for p in model.parameters()),
      "Qwen_weights_updated":False}

def terminal_reward(env,state):
    s=state.native
    assert s.done and not(s.success and s.violation)
    if s.violation:return -1.
    if s.success:return float(round(1-env.spec.cost_per_step*s.steps,6))
    return 0.

def terminal_proof(env,st):
    s=st.native
    assert s.done
    assert all(x["fact"] in s.delivered[x["recipient"]] for x in s.ledger)
    if s.success:
        assert s.generated_result and not s.violation
        assert s.business["verified_upstream_workflow"]["source_status"]=="delivered"
    return {
      "value":terminal_reward(env,st),
      "success":int(s.success and not s.violation),
      "violation":int(s.violation),
      "incomplete":int(not s.success and not s.violation),
      "steps":s.steps,
      "business_result":s.generated_result,
      "business_hash":digest(s.business),
      "R_delivered":list(s.delivered["R"]),
      "M_delivered":list(s.delivered["M"]),
      "actual_received_message_log":list(s.ledger),
      "native_hash":digest(asdict(s)),
    }

class ExactQpi:
    def __init__(self,env,model,ckpt_sha256,*,role_intervention=None):
        self.role_intervention=role_intervention # OFF by default; causal audit ONLY, never training Qpi
        self.env=env
        self.model=model
        self.ckpt=ckpt_sha256
        self.memo={}
        self.logits_cache={}
        self.count=Counter()
        self.stage_edges=Counter()
        self.edge_fingerprint=sha256()
        self.edge_examples=[]
        self.terminal_examples=[]

    def probs(self,obs,actions):
        key=digest({"observation":obs,"options":actions,"checkpoint":self.ckpt})
        if key not in self.logits_cache:
            with torch.no_grad():
                x=stack([actor_feature(obs,a) for a in actions])
                scores=self.model(x)
                assert torch.isfinite(scores).all()
                p=torch.softmax(scores.to(torch.float64),dim=0).tolist()
            assert all(z>0 and math.isfinite(z) for z in p)
            assert abs(sum(p)-1)<1e-12
            self.logits_cache[key]=p
            self.count["unique_frozen_actor_softmax_calls"]+=1
        return self.logits_cache[key]

    def walk(self,st,depth=0):
        assert depth<64,"Exceeded maximum documented action count"
        h=digest(asdict(st))
        if h in self.memo:
            self.count["memo_hits"]+=1
            return self.memo[h]
        self.count["native_states"]+=1
        if st.native.done:
            proof=terminal_proof(self.env,st)
            self.count["terminal_states"]+=1
            if len(self.terminal_examples)<20:self.terminal_examples.append(proof)
            ret=(proof["value"],float(proof["success"]),float(proof["violation"]),
                 float(proof["incomplete"]),float(proof["steps"]))
            self.memo[h]=ret
            return ret
        obs=self.env.observation(st)
        options=list(self.env.options(st))
        assert options
        actor=obs.get("actor",obs.get("role"))
        stage=obs.get("stage",obs.get("current_stage"))
        choices=valid_decision_candidates(obs,options)
        if len(choices)>=2:
            actions=choices;prob=self.probs(obs,actions)
            if self.role_intervention is not None:
                override=self.role_intervention(self.env,st,obs,actions)
                if override is not None:
                    assert override in actions
                    prob=[float(a==override) for a in actions]
                    self.count["privileged_oracle_role_intervention_not_training_Qpi"]+=1
            self.count["learned_actor_decisions"]+=1
        else:
            act=baseline_automatic_action(self.env,st)
            assert act in options,(actor,stage,options,act)
            actions=[act];prob=[1.0]
            self.count["fixed_adapter_decisions"]+=1
        sums=[0.,0.,0.,0.,0.]
        prior_hash=digest(asdict(st.native))
        prior_logs=list(st.native.ledger)
        for action,p in zip(actions,prob):
            child,audit=self.env.transition(st,action)
            assert audit["verdict"] in ("VALID","VIOLATION"),(actor,stage,action,audit)
            assert child.native.steps>st.native.steps or child.native.done
            assert digest(asdict(st.native))==prior_hash,"Native state mutated in parent branch"
            assert st.native.ledger==prior_logs,"Receiver history mutated in parent branch"
            assert child.native.ledger[:len(prior_logs)]==prior_logs
            for receipt in child.native.ledger[len(prior_logs):]:
                assert receipt["fact"] in child.native.delivered[receipt["recipient"]]
                assert receipt["sender"] and receipt["recipient"] in ("R","M")
            newhash=digest(asdict(child))
            e={
              "state":h,"native":prior_hash,"role":actor,"stage":stage,
              "action":action,"policy_prob":round(float(p),15),
              "verdict":audit["verdict"],"child":newhash,
              "child_native":digest(asdict(child.native)),
              "before_R_receipts":len(st.native.delivered["R"]),
              "after_R_receipts":len(child.native.delivered["R"]),
              "after_R_knowledge_hash":digest(child.native.delivered["R"]),
              "after_business_hash":digest(child.native.business),
            }
            self.edge_fingerprint.update((json.dumps(e,sort_keys=True,separators=(",",":"))+"\n").encode("utf-8"))
            self.count["actual_native_transition_edges"]+=1
            self.stage_edges[(actor,stage,action,audit["verdict"])]+=1
            if len(self.edge_examples)<40:self.edge_examples.append(e)
            value=self.walk(child,depth+1)
            for i,v in enumerate(value):sums[i]+=p*v
        assert abs(sum(sums[1:4])-1)<1e-7,(stage,sums)
        result=tuple(sums)
        self.memo[h]=result
        return result

    def evaluate_A_arms(self,source_state):
        assert source_state.phase=="A_SEND"
        source_obs=self.env.observation(source_state)
        assert source_obs["actor"]=="RecordsAgent"
        assert {"A_SEND_PACKAGE_A","A_SEND_PACKAGE_B"}<=set(source_obs["legal_actions"])
        origin=digest(asdict(source_state))
        arms={}
        for choice in ("PACKAGE_A","PACKAGE_B"):
            delivered,audit=self.env.transition(source_state,"A_SEND_"+choice)
            assert audit["verdict"]=="VALID" and not delivered.native.violation
            assert len(delivered.native.delivered["R"])>0
            assert all(z["sender"]=="RecordsAgent" and z["recipient"]=="R"
                       for z in delivered.native.ledger)
            q,ok,leak,incomplete,steps=self.walk(delivered)
            arms[choice]={
              "Qpi":round(q,9),
              "success_probability":round(ok,9),
              "CI_violation_probability":round(leak,9),
              "incomplete_probability":round(incomplete,9),
              "expected_native_steps":round(steps,9),
              "forced_A_action_verdict":audit["verdict"],
              "actual_A_delivered_ledger":delivered.native.ledger,
              "actual_A_delivered_R":delivered.native.delivered["R"],
              "A_successor_native_state_sha256":digest(asdict(delivered.native)),
            }
        assert digest(asdict(source_state))==origin
        return {
          "source_chain_hash":origin,
          "source_actor_observation_sha256":digest(source_obs),
          "signed_source_tool_receipt_sha256":source_state.source_certificate,
          "A_role":"RecordsAgent","phase":"A_SEND",
          "arms":arms,
          "Qpi_A_minus_B":round(arms["PACKAGE_A"]["Qpi"]-arms["PACKAGE_B"]["Qpi"],9),
          "cost":{
            **dict(self.count),
            "policy_softmax_cache_entries":len(self.logits_cache),
            "edge_sha256":self.edge_fingerprint.hexdigest(),
            "per_actor_stage_action_edges":{
               "|".join(str(x) for x in k):v for k,v in sorted(self.stage_edges.items())},
          },
          "native_edge_proof_examples":self.edge_examples,
          "terminal_proof_examples":self.terminal_examples,
        }

def prepare_A(env):
    s=env.initial()
    assert not s.native.ledger and not s.native.delivered["R"]
    s,r=env.transition(s,"A_READ_VERIFIED_PACKAGES")
    assert r["verdict"]=="VALID" and s.phase=="A_SEND"
    return s

def qstar_A_from_original(row,verify=False):
    values=row["upstream_A_values_given_exact_B_continuation"]
    assert set(values)=={"PACKAGE_A","PACKAGE_B"}
    if verify:
        env=environment(row)
        for pkg in values:
            s=prepare_A(env)
            s,a=env.transition(s,"A_SEND_"+pkg)
            assert a["verdict"]=="VALID"
            for act in ("INSPECT_BUSINESS","CHOOSE:"+env.native._route):
                s,a=env.transition(s,act)
                assert a["verdict"]=="VALID"
            vmax=max(native_best(env.native,s.native,act)["native_exact_Qstar"]
                 for act in ("OPEN_PRIVATE_CHANNEL","SKIP_PRIVATE_CHANNEL"))
            assert abs(vmax-values[pkg])<1e-7,(row["world"],pkg,vmax,values[pkg])
    return values
