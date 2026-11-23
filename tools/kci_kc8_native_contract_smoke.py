"""Exact KC8 TRAIN actor prompt/state compatibility smoke. NO LLM."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_kc8_real4b_train_diagnostic import (load_native_snapshot,clone_prefix,D)
from kci_pisas_model_side_bridge import (canonical_actor_candidates,semantic_candidate_prompt,current_model_entrypoints)
def main():
    candidates,CausalEnv,model_only,digest,verifier=load_native_snapshot()
    cases={c.id:c for c in candidates() if c.split=="train" and c.id.startswith("KC8")}
    assert len(cases)==6
    data=D/"FROZEN_KC8_TRAIN_INPUTS/MODEL_ONLY_TRAIN_A.jsonl"
    rows=[json.loads(x) for x in data.open(encoding="utf8")]
    assert len(rows)==12
    prompt_ids={r["prompt_sha256"] for r in rows}
    assert len(prompt_ids)==12
    matched=0
    for c in cases.values():
      for h in ("H1","H2"):
        env=CausalEnv(c,h)
        st=clone_prefix(env)
        ob=env.actor_observation(st)
        view=model_only(ob)
        assert view["prompt_sha256"] in prompt_ids, (c.id,h,view["prompt_sha256"])
        acts,descs,txt=canonical_actor_candidates(view)
        assert acts==("ABORT","SEND_A","SEND_B")
        assert "ACTUALLY RECEIVED" in txt
        assert "Qstar" not in txt and "native_teacher" not in txt
        for a in acts:
            p=semantic_candidate_prompt(view,a)
            assert "Candidate action under consideration: "+a in p
            assert "Answer A if" in p
        matched+=1
    assert matched==12
    entries=current_model_entrypoints()
    assert "HFActionPolicy" in entries and "DPR_repair" in entries
    print("KC8_TRAIN_ACTOR_PROMPT_NATIVE_SMOKE_PASS",matched,
      "exact_hash_matches, roles/ACL preserved",len(entries),"pre-existing method components")
if __name__=="__main__":main()
