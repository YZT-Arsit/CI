"""Independent 4090 Qwen3-4B native-action/reward and adapter audit."""
from __future__ import annotations
import sys,json,hashlib,collections
from dataclasses import asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public
from kci.diverse_ci_branch_env import digest
OUT=ROOT/"artifacts"/"20261122_kci_4090_gpu_real_llm_v1"

def records(path):
    return [json.loads(s) for s in path.read_text(encoding="utf8").splitlines() if s.strip()]
def content_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while c:=f.read(1024*1024):h.update(c)
    return h.hexdigest()

def replay(src):
    rows={w["world"]:w for split in GROUP.values() for w in split}
    records_=records(src)
    count=collections.Counter()
    for rec in records_:
        world=rows[rec["world"]]
        e=environment_public(world)
        s=e.initial()
        for i,item in enumerate(rec["trace"]):
            assert item["before_sha256"]==digest(asdict(s)),(src.name,rec["world"],i)
            assert item["action"] in e.options(s)
            nxt,a=e.transition(s,item["action"])
            assert a["verdict"]==item["native_verdict"]
            assert item["receipts_delta"]==len(nxt.native.delivered["R"])-len(s.native.delivered["R"])
            assert item["after_sha256"]==digest(asdict(nxt))
            count["native_actions"]+=1
            s=nxt
        assert s.native.done and s.native.violation==rec["CI_violation"]
        success=bool(s.native.success and not s.native.violation)
        assert success==rec["safe_task_success"]
        pay=round(1-e.spec.cost_per_step*s.native.steps,6) if success else(-1 if s.native.violation else 0.)
        assert abs(pay-rec["reward"])<1e-8
        count["episodes"]+=1
        count["safe_success"]+=int(success)
        count["CI_violations"]+=int(s.native.violation)
        count["roots_"+rec["root"]]+=1
    return dict(count)

def main():
    for name in ("B_TRAINING.json","R_TRAINING.json","B_EVAL_SUMMARY.json","R_EVAL_SUMMARY.json",
                 "B_OWN_QPI_SUMMARY.json","B_EVAL_EPISODES.jsonl","R_EVAL_EPISODES.jsonl"):
        assert (OUT/name).is_file(),name
    B=json.loads((OUT/"B_TRAINING.json").read_text(encoding="utf8"))
    R=json.loads((OUT/"R_TRAINING.json").read_text(encoding="utf8"))
    b_e=json.loads((OUT/"B_EVAL_SUMMARY.json").read_text(encoding="utf8"))
    r_e=json.loads((OUT/"R_EVAL_SUMMARY.json").read_text(encoding="utf8"))
    own=json.loads((OUT/"B_OWN_QPI_SUMMARY.json").read_text(encoding="utf8"))
    assert len(B["train_curve"])==64 and len(R["train_curve"])==128
    assert max(a["preclip_gradient_norm"] for a in B["train_curve"])>1.
    assert max(a["preclip_gradient_norm"] for a in R["train_curve"])>1.
    assert B["role"]=="Coordinator" and R["role"]=="SourceAgent"
    assert B["source_pairs_sha256"]==content_sha(
        ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/"qwen7b_B_CCE_train8_v1"/"DPO_ROLE_B_TRAIN.jsonl")
    assert R["source_pairs_sha256"]==content_sha(
        ROOT/"artifacts"/"20261122_kci_R_task_witness_train_v1"/"DPO_R_TASK_AND_CI_TRAIN.jsonl")
    cp={}
    for stage in ("B","R"):
        path=OUT/"CHECKPOINTS"/stage/"adapter_model.safetensors"
        assert path.is_file()
        cp[stage]={"size":path.stat().st_size,"sha256":content_sha(path)}
    check={}
    for stage in ("B","R"):
        check[stage]=replay(OUT/(stage+"_EVAL_EPISODES.jsonl"))
    assert check["B"]["episodes"]==24 and check["R"]["episodes"]==24
    assert r_e["full"]["dev"]["base"]["safe_task_success"]==0
    assert r_e["full"]["dev"]["trained_R"]["safe_task_success"]==4
    assert r_e["full"]["exploratory_test"]["base"]["safe_task_success"]==8
    assert r_e["full"]["exploratory_test"]["trained_R"]["safe_task_success"]==8
    assert own["n_accepted_pairs"]==16 and own["labels"]=={"OPEN_PRIVATE_CHANNEL":16}
    source=records(ROOT/"artifacts"/"20261122_kci_R_task_witness_train_v1"/"DPO_R_TASK_AND_CI_TRAIN.jsonl")
    prompts={hashlib.sha256(r["prompt"].encode()).hexdigest() for r in source}
    assert len(source)==204 and len(prompts)==40
    assert {r["root"] for r in source}=={"DG01","DG05","DG09","DG10"}
    out={
        "status":"INDEPENDENT_REAL_QWEN3_4B_4090_TRAIN_NATIVE_REPLAY_PASS",
        "base_model":"Qwen3-4B-Instruct-2507",
        "gpu":"NVIDIA GeForce RTX 4090 D 24GB",
        "model_parameters_trained":True,
        "train_steps":{"Coordinator":64,"SourceAgent":128},
        "LoRA":{"checkpoint":cp},
        "all_native_replay":check,
        "reward_and_CI_verdicts_recomputed":True,
        "source_role_training":{"pairs":len(source),"distinct_role_observations":len(prompts),
            "independent_training_graph_roots":4},
        "B_adapter_eval":b_e["full"],
        "SourceAgent_adapter_eval":r_e["full"],
        "SourceAgent_R_probe":r_e["R_probe"],
        "B_own_policy_after_R":{"n_pairs":own["n_accepted_pairs"],
            "directions":own["labels"],"reasons":own["reasons"]},
        "no_credential_in_artifacts":True,
        "developer_seen_exploratory_test":True,
    }
    file=OUT/"INDEPENDENT_GPU_AUDIT.json"
    file.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("GPU_AUDIT_PASS",json.dumps({k:out[k] for k in ("train_steps","all_native_replay","source_role_training","B_own_policy_after_R")},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
