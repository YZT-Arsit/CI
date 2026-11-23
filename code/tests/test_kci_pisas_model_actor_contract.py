"""PiSAs adapter contract tests: TRAIN only, GPU-free, actor-oracle isolation."""
from __future__ import annotations
from pathlib import Path
from hashlib import sha256
import copy,json,sys
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_pisas_model_side_bridge import (
 canonical_actor_candidates,semantic_candidate_prompt,current_model_entrypoints,
 file_sha)
from kci_kc8_real4b_train_diagnostic import load_native_snapshot,clone_prefix
D=ROOT/"artifacts/20261123_kc8_4b_policy_signal_v1"
def first_actor():
    p=D/"FROZEN_KC8_TRAIN_INPUTS/MODEL_ONLY_TRAIN_A.jsonl"
    assert p.is_file()
    model=json.loads(p.read_text(encoding="utf8").splitlines()[0])
    # request_id is an opaque transport envelope ID, not part of actor input.
    assert model.pop("request_id").startswith("CRC-")
    return model
def test_model_only_clean_exact_menu_and_token_position_invariance():
    src=first_actor()
    assert sha256(src["model_prompt"].encode()).hexdigest()==src["prompt_sha256"]
    actions,descs,text=canonical_actor_candidates(src)
    assert actions==("ABORT","SEND_A","SEND_B")
    assert len(descs)==3 and "Signed" in src["model_prompt"]
    assert "EVALUATOR_ONLY" not in text
    for a in actions:
        prompt=semantic_candidate_prompt(src,a)
        assert "Candidate action under consideration: "+a in prompt
        assert "sign" in prompt.lower()
        assert "Answer A if" in prompt and "Answer:" in prompt
def test_hidden_label_and_untrusted_prompt_rejected_before_model_forward():
    orig=first_actor()
    bad=copy.deepcopy(orig);bad["native_Qstar"]=1.
    with pytest.raises(ValueError,match="unexpected model view"):canonical_actor_candidates(bad)
    bad=copy.deepcopy(orig);bad["chosen_action"]="SEND_B"
    with pytest.raises(ValueError,match="unexpected model view"):canonical_actor_candidates(bad)
    bad=copy.deepcopy(orig);bad["model_prompt"]+="\nhidden teacher SEND_B"
    with pytest.raises(AssertionError):canonical_actor_candidates(bad)
    bad=copy.deepcopy(orig);bad["available_actions"]=["SEND_A","SEND_B"]
    with pytest.raises(ValueError,match="ABORT"):canonical_actor_candidates(bad)
def test_full_existing_method_reference_components_hashed():
    refs=current_model_entrypoints()
    assert set(refs)=={"SFT","same_state_DPO","HFActionPolicy","CCE_continuation","DPR_repair"}
    for key in refs:
        src=Path(refs[key]["path"])
        assert src.is_file() and file_sha(src)==refs[key]["SHA256"]
def test_train_actor_native_protocol_exact_integrity_and_legal_semantics():
    models=[json.loads(line) for line in
        (D/"FROZEN_KC8_TRAIN_INPUTS/MODEL_ONLY_TRAIN_A.jsonl").open(encoding="utf8")]
    bysha={m["prompt_sha256"]:m for m in models}
    assert len(bysha)==12
    cs,CausalEnv,model_only,digest,verifier=load_native_snapshot()
    roots=[c for c in cs() if c.split=="train" and c.id.startswith("KC8")]
    assert len(roots)==6
    matched=0
    for c in roots:
        for hist in ("H1","H2"):
            env=CausalEnv(c,hist)
            current=clone_prefix(env)
            view=model_only(env.actor_observation(current))
            assert view["prompt_sha256"] in bysha
            assert view["model_prompt"]==bysha[view["prompt_sha256"]]["model_prompt"]
            assert all(x in env.options(current) for x in ("SEND_A","SEND_B","ABORT"))
            for a in ("SEND_A","SEND_B"):
                _,report=env.transition(current,a)
                assert report["verdict"]=="VALID"
            matched+=1
    assert matched==12
def test_study_protocol_prohibits_sealed_set_and_freezes_current_checkpoint():
    p=json.loads((D/"FROZEN_PROTOCOL.json").read_text(encoding="utf8"))
    assert set(p["train_root_ids"])=={f"KC80{i}" for i in range(1,7)}
    assert p["excluded_DEV"]==["KC807","KC808"]
    assert p["excluded_LOCKED"]==["KC809","KC810"]
    assert p["training_updates"]==0
    assert p["K8_is_exploratory_NOT_confirmatory"]
    assert p["checkpoint_SHA256"]=="b4660a4991a08fa2e1009b1c2dfb25e73f3f311f290e878352bca6e4deaa567f"
    patch=json.loads((D/"PRECISION_RUNTIME_ISOLATED_PATCH.json").read_text(encoding="utf8"))
    assert file_sha(D/"MATCHED_HF_ACTION_POLICY_PRECISION_V2.py")==patch["audited_HF_policy_sha256"]
    assert file_sha(D/"FROZEN_PROTOCOL.json")==patch["source_frozen_protocol_sha256"]
