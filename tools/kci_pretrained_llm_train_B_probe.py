"""Measure actual Qwen7B B choices on TRAIN prefixes independently of fork labels."""
from __future__ import annotations
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_pretrained_llm_train_forks import to_preflight
from kci_pretrained_llm_native_runner import OllamaActor

def main():
    existing=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/"qwen7b_A_all28_v1"/"EPISODES.jsonl"
    data={r["world"]:r for r in (json.loads(x) for x in existing.read_text(encoding="utf8").splitlines())}
    worlds=[w for w in GROUP["train"] if not w["swapped_package_names"]]
    actor=OllamaActor("qwen2.5:7b-instruct-q4_K_M")
    out=ROOT/"artifacts"/"20261122_kci_pretrained_llm_native"/"qwen7b_B_train_policy_probe_v1"
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    records=[]
    for w in worlds:
        e,s,prefix=to_preflight(w,data[w["world"]]["action"])
        obs=e.observation(s)
        action,raw,usage,hashvalue=actor.choose(obs)
        r={"world":w["world"],"root":w["root"],"split":"train",
           "action":action,"model_response":raw,"usage":usage,
           "source_A_model_choice":data[w["world"]]["action"],
           "preflight_observation_sha256":hashvalue}
        records.append(r)
        print("REAL_LLM_TRAIN_B",w["world"],action,usage,flush=True)
    summary={"status":"ACTUAL_FROZEN_7B_B_BEHAVIOR_ON_TRAIN_PREFIX",
        "n":len(records),"n_OPEN":sum(x["action"]=="OPEN_PRIVATE_CHANNEL" for x in records),
        "n_SKIP":sum(x["action"]=="SKIP_PRIVATE_CHANNEL" for x in records),
        "invalid":sum(x["action"] is None for x in records),
        "independent_roots":len(set(x["root"] for x in records)),
        "model_unchanged":True}
    (out/"B_ACTIONS.json").write_text(json.dumps({"summary":summary,"cases":records},indent=2)+"\n",encoding="utf8")
    print("SUMMARY",json.dumps(summary),flush=True)
if __name__=="__main__":main()
