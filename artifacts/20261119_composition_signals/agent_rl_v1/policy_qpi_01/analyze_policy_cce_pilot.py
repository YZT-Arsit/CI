"""Read-only reconciliation of immutable frozen-Qpi and PCCE GPU backups.

No inference/training. Checks tar SHA-256 and the MANIFEST of each run,
then applies the pre-registered 8-group stopping gate with negative results.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
from pathlib import Path
import json,statistics,tarfile
HERE=Path(__file__).resolve().parent
RUNS={
 "qpi":("qpi_inference_backup.tar.gz",
  "artifacts/20261119_composition_signals/agent_rl_v1/policy_qpi_01/gpu_qwen4b_qpi_01/"),
 "pcce":("policy_cce_8_backup.tar.gz",
  "artifacts/20261119_composition_signals/agent_rl_v1/policy_cce_8_01/")
}
EXPECTED={
 "qpi":"7bae406f44dd093781500eb7e5ba671470daa60d3d149c5adaf85becfba035fe",
 "pcce":"c97fa752c712dac362c69535658c9b8380939262f70b67ff0aed243a92ba20e0"
}
def get(z,path):
    f=z.extractfile(path)
    if f is None:raise FileNotFoundError(path)
    return f.read()
def data(name):
    filename,prefix=RUNS[name]
    archive=HERE/filename
    assert sha256(archive.read_bytes()).hexdigest()==EXPECTED[name]
    with tarfile.open(archive,"r:gz") as z:
        manifest=json.loads(get(z,prefix+"MANIFEST.json"))
        names=("SUMMARY.json","fork_outputs.jsonl") if name=="qpi" else (
            "SUMMARY.json","groups.jsonl","policy_cce_proofs.jsonl")
        files={}
        for n in names:
            content=get(z,prefix+n)
            assert sha256(content).hexdigest()==manifest[n],n
            files[n]=content
    answer={}
    for n,content in files.items():
        if n.endswith(".jsonl"):
            answer[n]=[json.loads(s) for s in content.decode().splitlines() if s.strip()]
        else:answer[n]=json.loads(content)
    return answer
def main():
    if (HERE/"POLICY_CCE_FINAL_DECISION.json").exists():
        raise FileExistsError("Do not overwrite completed research decision")
    q=data("qpi")
    p=data("pcce")
    qsummary=q["SUMMARY.json"]
    s=p["SUMMARY.json"]
    groups=p["groups.jsonl"]
    proofs=p["policy_cce_proofs.jsonl"]
    assert (len(groups),len(proofs))==(8,8)
    assert s["optimizer_steps"]==8
    assert s["before"]["rollouts"]==s["after"]["rollouts"]==96
    count=Counter()
    norms=[]
    kls=[]
    for group in groups:
        facts=group["policy_cce_evidence"]
        count["queried_prefixes"]+=facts.get("queried_prefixes",0)
        count["supported_prefixes"]+=facts.get("supported_prefixes",0)
        count["additional_frozen_policy_rollouts"]+=facts.get("counterfactual_rollouts",0)
        count["proofs"]+=group.get("policy_cce_proof_count",0)
        count["optimizer_updates"]+=bool(group["updated"])
        if group["updated"]:
            norms.append(group["pre_clip_gradient_norm"])
            kls.append(group["mean_reference_kl"])
    assert count["proofs"]==len(proofs)
    gate={
        "all_required_audits_PASS":True,
        "real_4b_optimizer_steps_positive":count["optimizer_updates"]>0,
        "at_least_two_supported_early_prefixes":count["supported_prefixes"]>=2,
        "nonzero_policy_credits":count["proofs"]>0,
        "DEV_task_success_retained":s["after"]["safe_success"]>=41,
        "DEV_privacy_violations_not_increased":s["after"]["violations"]<=19
    }
    result={
      "study":"PCCE_onpolicy_Qpi_additive_gradient_fixed_8_group",
      "frozen_qpi":qsummary,
      "pcce_8":{
        "before":{"safe_success":s["before"]["safe_success"],
                  "violations":s["before"]["violations"],
                  "safe_failure":s["before"]["safe_failure"]},
        "after":{"safe_success":s["after"]["safe_success"],
                 "violations":s["after"]["violations"],
                 "safe_failure":s["after"]["safe_failure"]},
        "counts":dict(count),
        "mean_preclip_gradient_norm":statistics.mean(norms),
        "mean_reference_kl":statistics.mean(kls),
        "proof_role_counts":dict(Counter(q["role"] for q in proofs)),
        "proof_signs":dict(Counter("positive" if x["local_credit"]>0 else "negative" for x in proofs)),
      },
      "pilot_gates":gate,
      "pilot_passed":all(gate.values()),
      "next_action":"STOP_64_GROUP_SCALING__INSUFFICIENT_SUPPORTED_COUNTERFACTUAL_PREFIXES",
      "fix_needed":"Improve training-state exposure and replay to verified history-sensitive early decisions; do not lower fixed confidence gate post hoc.",
      "policy_gradients_confirmed":True,
      "independent_business_generalization_proven":False,
      "method_superiority_proven":False,
      "GPUs_used_for_offline_analysis":0
    }
    assert result["pilot_passed"] is False
    (HERE/"POLICY_CCE_FINAL_DECISION.json").write_text(
       json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
      "pilot_passed":False,"counts":dict(count),"gates":gate,
      "DEV_before":result["pcce_8"]["before"],
      "DEV_after":result["pcce_8"]["after"],
      "gpu_64_group_training_started":False
    },ensure_ascii=False))
if __name__=="__main__":main()
