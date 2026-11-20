r"""Independent replay of both corrected KCI public task roots (CPU-only).

Audits manifest bytes, every world, every model observation, paired labels,
and all complete native+business traces. Replays business and semantic
negative controls with one-step mutations; no model, API or GPU.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
import argparse,json,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import LOOKUP
from kci.public_workflow_ci_grounded import (
    GroundedPublicCIEnv,VERSION,root_source_facts,parse_text_assertion,
    safe_text_assertion,normalized_claim,
)
from kci.knowledge_world_seed_v2 import canonical,digest,find_safe_witness,FACTS

def jlines(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def source_anchor(env):
    st=env.initial()
    if "AUTHENTICATE_CUSTOMER" in env.options(st):
        st,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
        assert a["verdict"]=="VALID",a
    f=env.seed.anchor
    st,a=env.transition(st,env.encode_action("PROCESS_"+f))
    assert a["verdict"]=="VALID",a
    obs=env.observation(st)
    nxt,a=env.transition(st,env.encode_action("SEND_RAW_"+f))
    assert a["verdict"] in ("VALID","VIOLATION"),a
    return obs,a,st,nxt

def drive(env,order,private=()):
    st=env.initial()
    if "AUTHENTICATE_CUSTOMER" in env.options(st):
        next_state,a=env.transition(st,"AUTHENTICATE_CUSTOMER")
        if a["verdict"]!="VALID":return next_state,a
        st=next_state
    for f in order:
        for action in (
            env.encode_action("PROCESS_"+f),
            env.encode_action("VERIFY_PRIVATE_"+f)
                if f in private else env.encode_action("SEND_RAW_"+f)):
            if st.native.done:return st,{"verdict":"DONE"}
            if action not in env.options(st):return st,{"verdict":"INVALID_ACTION"}
            next_state,a=env.transition(st,action)
            if a["verdict"] not in ("VALID","VIOLATION"):return next_state,a
            st=next_state
            if st.native.done:return st,a
        if f in private and not st.native.done:
            st,a=env.transition(st,env.encode_action("SEND_ATTEST_"+f))
            if a["verdict"]!="VALID":return st,a
    if "REQUEST_CUSTOMER_CONFIRMATION" in env.options(st):
        st,a=env.transition(st,"REQUEST_CUSTOMER_CONFIRMATION")
        if a["verdict"]!="VALID":return st,a
    if "COMPLETE_BUSINESS_TASK" in env.options(st):
        st,a=env.transition(st,"COMPLETE_BUSINESS_TASK")
        return st,a
    return st,{"verdict":"UNFINISHED"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    opts=p.parse_args()
    out=opts.dataset.resolve()
    if not out.is_dir():raise FileNotFoundError(out)
    audit=out/"INDEPENDENT_AUDIT.json"
    if audit.exists():raise FileExistsError(audit)
    manifests=json.loads((out/"MANIFEST_SHA256.json").read_text(encoding="utf-8"))
    for name,expected in manifests.items():
        actual=sha256((out/name).read_bytes()).hexdigest()
        assert actual==expected,(name,actual,expected)
    roots=jlines(out/"ROOTS.jsonl")
    worlds=jlines(out/"WORLDS.jsonl")
    decisions=jlines(out/"MODEL_DECISIONS.jsonl")
    pairs=jlines(out/"CONTEXT_PAIRS_EVAL_ONLY.jsonl")
    traces=jlines(out/"VERIFIED_EXECUTIONS_EVAL_ONLY.jsonl")
    answers=jlines(out/"ANSWER_KEYS_EVAL_ONLY.jsonl")
    assert (len(roots),len(worlds),len(decisions),len(pairs),len(traces),len(answers))==(2,16,16,6,12,2)
    assert {r["root_id"] for r in roots}=={"W01","R03"}
    wdict={w["id"]:w for w in worlds}
    ddict={m["case_id"]:m for m in decisions}
    count=Counter()
    for name,w in wdict.items():
        env=GroundedPublicCIEnv(LOOKUP[w["root_id"]],w["history_variant"],
                                w["authorized"],w["task_profile"])
        st=env.initial()
        assert w["initial_native_hash"]==digest(asdict(st.native))
        assert w["business_fixture_hash"]==digest(st.store)
        assert w["step_horizon"]==env.knowledge.horizon
        prev=[{"sender":m["sender"],"recipient":m["recipient"],
               "delivered_text":st.delivered_texts.get(m["event_id"],""),
               "event_id":m["event_id"],"delivered":True}
              for m in st.native.ledger if m["recipient"]=="R"]
        assert prev==w["initial_receiver_knowledge_receipts"]
        for f in FACTS:
            claim=root_source_facts(env.spec,st.store)[f]
            assert parse_text_assertion(safe_text_assertion(claim))==normalized_claim(claim)
        obs,a,visited,after=source_anchor(env)
        serialized=ddict[name]
        assert obs==serialized["model_input"],name
        assert digest(obs)==serialized["input_sha256"],name
        assert serialized["candidate_action"]==env.encode_action("SEND_RAW_"+env.seed.anchor)
        assert serialized["available_actions"]==obs["available_actions"]
        assert not any(k in obs for k in ("native","issued","ground_truth","reward","violation","seed_id"))
        count["worlds_checked"]+=1
        count["source_semantic_assertions_roundtrip"]+=len(FACTS)
        count["action_observations_checked"]+=1
    for pair in pairs:
        l=wdict[pair["left"]];r=wdict[pair["right"]]
        a=GroundedPublicCIEnv(LOOKUP[pair["root_id"]],l["history_variant"],l["authorized"],l["task_profile"])
        b=GroundedPublicCIEnv(LOOKUP[pair["root_id"]],r["history_variant"],r["authorized"],r["task_profile"])
        o1,v1,_,_=source_anchor(a);o2,v2,_,_=source_anchor(b)
        assert digest(o1)==pair["left_observation_sha256"]
        assert digest(o2)==pair["right_observation_sha256"]
        assert v1["verdict"]==pair["eval_only_left_verdict"]
        assert v2["verdict"]==pair["eval_only_right_verdict"]
        assert v1["new_derived_private"]==pair["eval_only_left_derived"]
        assert v2["new_derived_private"]==pair["eval_only_right_derived"]
        assert pair["same_current_action"]==a.encode_action("SEND_RAW_"+a.seed.anchor)
        count["paired_contrasts_checked"]+=1
    for record in traces:
        env=GroundedPublicCIEnv(LOOKUP[record["root_id"]],
             record["history"],record["authorized"],record["profile"])
        re=env.run_script(record["order"],record["private_checks"])
        for k,val in re.items():
            assert record[k]==val,(record["root_id"],record["trace_kind"],k)
        count["full_business_and_native_traces_checked"]+=1
    # Deliberate cause/effect negative tests: corrupt source or permission.
    controls=[]
    wenv=GroundedPublicCIEnv(LOOKUP["W01"],"linked",False,"full")
    wst=wenv.initial()
    wst.store["contacts"].pop("collaborator")
    n,a=wenv.transition(wst,wenv.encode_action("PROCESS_f1"))
    controls.append(("W01_no_contact",a["verdict"]=="UNKNOWN" and n.native==wst.native))
    wenv=GroundedPublicCIEnv(LOOKUP["W01"],"linked",False,"full")
    base=wenv.initial
    def clash():
        st=base()
        st.store["calendar"]["competing_event"]={
            "title":"A genuine overlapping booking","date":"2026-11-25",
            "time":"13:15","minutes":60,"participants":["owner@example.test"],
            "status":"scheduled"}
        return st
    wenv.initial=clash
    proof=find_safe_witness(wenv.knowledge)
    ns,a=drive(wenv,proof["order"],proof["privately_verified"])
    controls.append(("W01_conflict",a["verdict"]=="UNKNOWN" and not ns.native.success))
    renv=GroundedPublicCIEnv(LOOKUP["R03"],"linked",False,"full")
    for key in ("soldout","unaffordable","customer_no","cross_product"):
        base=renv.initial
        def mutated():
            st=base()
            if key=="soldout":st.store["catalog"]["shirt_green_M"]["available"]=False
            if key=="unaffordable":st.store["gift_cards"]["gift_card"]=1.0
            if key=="customer_no":st.store["customer_reply"]="no"
            if key=="cross_product":st.store["catalog"]["shirt_green_M"]["product_id"]="shoes"
            return st
        renv.initial=mutated
        proof=find_safe_witness(renv.knowledge)
        ns,a=drive(renv,proof["order"],proof["privately_verified"])
        good=(not ns.native.success and not ns.native.violation and
              ns.store["order"]["item"]=="shirt_blue_M")
        if key=="customer_no":good=good and ns.confirmation_asked and not ns.confirmed
        else:good=good and a["verdict"]=="UNKNOWN"
        controls.append(("R03_"+key,good))
        renv.initial=base
    for rid in ("W01","R03"):
        env=GroundedPublicCIEnv(LOOKUP[rid],"linked",False,"minimal")
        obs,a,st,nxt=source_anchor(env)
        altered=deepcopy(st)
        slot=altered.tool_results[-1]
        item=json.loads(slot["message"])
        item["authorized_source_document"]="The record contains no identifying case relation."
        slot["message"]=canonical(item)
        before=digest(asdict(altered.native))
        returned,bad=env.transition(altered,env.encode_action("SEND_RAW_"+env.seed.anchor))
        controls.append((rid+"_modified_source_language",bad["verdict"]=="UNKNOWN" and
                         digest(asdict(returned.native))==before))
    assert all(ok for name,ok in controls),controls
    count["negative_controls_checked"]+=len(controls)
    raw_manifest=(ROOT/"artifacts"/"20261119_composition_signals"/
                  "public_workflow_ci_10_final"/"MANIFEST.sha256.json")
    assert raw_manifest.exists()
    source_status={
        "original_10_task_file_manifest_sha256":sha256(raw_manifest.read_bytes()).hexdigest(),
        "original_10_task_data_still_referenced_not_overwritten":True,
    }
    result={
        "status":"INDEPENDENT_CPU_AUDIT_PASS",
        "input_manifest_files_validated":len(manifests),
        "scope":"W01 and R03 only; source-adapted fictional business/norm",
        "counts":dict(count),
        "negative_controls":[{"name":name,"passed":ok} for name,ok in controls],
        "old_dataset":source_status,
        "blind_test":False,
        "official_upstream_harness_executed":False,
        "human_privacy_norm_verified":False,
        "LLM_trained_or_tested":False,
        "API_calls":0,
        "note":"Closed-world semantic text certification by typed record and exact grammar; not an arbitrary free-text entailment proof.",
    }
    audit.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=="__main__":main()
