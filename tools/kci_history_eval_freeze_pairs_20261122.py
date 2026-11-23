"""Freeze matched role-visible inputs for evaluator A/B; no original traces changed."""
from __future__ import annotations
from pathlib import Path
from collections import Counter,defaultdict
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_history_eval_dev_probe_20261122 import OUT,at_first_release
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public
from kci_llm_text_surface import render_role_text
from kci.diverse_ci_branch_env import digest
RAW,VERIFY="SEND_RAW_R","VERIFY_PRIVATE_R"

def main():
    rows=[json.loads(s) for s in (OUT/"PAIR_STATES.jsonl").read_text(encoding="utf8").splitlines()]
    worlds={w["world"]:w for rr in GROUP.values() for w in rr}
    assert len(rows)==56 and len(worlds)==28
    by_teacher={"visible":[],"blind":[]}
    test=[]
    for row in rows:
        env=environment_public(worlds[row["world"]])
        st=at_first_release(env,row["upstream_package"])
        obs=env.observation(st)
        prompt=render_role_text(obs)
        assert prompt==row["visible_prompt"]
        assert digest(obs)==row["visible_observation_sha256"]
        assert sha256(prompt.encode()).hexdigest()==row["visible_prompt_sha256"]
        options=list(obs["legal_actions"])
        assert RAW in options and VERIFY in options
        assert options.index(RAW)!=options.index(VERIFY)
        def pair(teacher):
            label=row["aware_label"] if teacher=="visible" else row["blind_label"]
            rejected=VERIFY if label==RAW else RAW
            branch=row["aware"] if teacher=="visible" else row["blind"]
            return {
                "split":row["split"],"world":row["world"],"root":row["root"],
                "branch":row["branch"],"role":obs.get("actor",obs.get("role")),
                "stage":obs.get("current_stage"),
                "source":"native_verified_first_release_hist_state_ablation",
                "teacher_history_access":teacher,
                "state_hash":row["native_state_sha256"],
                "observation_hash":row["visible_observation_sha256"],
                "all_executable_actions":options,
                "prompt":prompt,
                "chosen_action":label,
                "rejected_action":rejected,
                "chosen_letter":chr(65+options.index(label)),
                "rejected_letter":chr(65+options.index(rejected)),
                "native_evidence":{a:{
                    "verdict":branch[a]["verdict"],
                    "safe_success":branch[a]["safe_success"],
                    "steps":branch[a]["steps"]
                } for a in (RAW,VERIFY)},
                "not_Qstar_oracle_teacher":True,
                "teacher_is_native_verifier_not_Qpi":True,
                "first_release_only":True,
                "actual_reference_label":row["aware_label"]
            }
        if row["split"]=="train":
            for teacher in by_teacher:
                p=pair(teacher)
                assert p["role"] in ("DirectoryAgent","OrderAgent","RecordsAgent","ComplianceAgent")
                by_teacher[teacher].append(p)
        else:
            p=pair("visible")
            test.append(p)
    assert len(by_teacher["visible"])==len(by_teacher["blind"])==32
    assert len(test)==24
    assert all((a["world"],a["branch"],a["prompt"],a["all_executable_actions"])==
          (b["world"],b["branch"],b["prompt"],b["all_executable_actions"])
          for a,b in zip(by_teacher["visible"],by_teacher["blind"]))
    assert len({x["root"] for x in by_teacher["visible"]})==4
    for teacher,items in by_teacher.items():
        p=OUT/("TRAIN_"+teacher.upper()+".jsonl")
        assert not p.exists(),p
        p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in items),encoding="utf8")
    p=OUT/"EVAL_GROUPED.jsonl"
    assert not p.exists(),p
    p.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in test),encoding="utf8")
    meta={"train_state_rows_per_arm":32,"train_independent_roots":4,
          "train_labels":{k:dict(Counter(x["chosen_action"] for x in v)) for k,v in by_teacher.items()},
          "train_root_labels":{k:{r:dict(Counter(x["chosen_action"] for x in v if x["root"]==r))
              for r in sorted({x["root"] for x in v})} for k,v in by_teacher.items()},
          "eval_rows":len(test),"eval_roots":sorted({x["root"] for x in test}),
          "train_hashes":{k:sha256((OUT/("TRAIN_"+k.upper()+".jsonl")).read_bytes()).hexdigest() for k in by_teacher},
          "paired_actor_inputs_byte_identical":True,
          "label_source":"native first-release replay with/without recipient history, fixed safe continuation"}
    q=OUT/"DATA_MANIFEST.json"
    assert not q.exists()
    q.write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(meta,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
