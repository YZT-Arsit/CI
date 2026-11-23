"""Qwen3-4B model-side role-action interface for native PiSAs/KC8 adapters.

This is a THIN reused semantic action scorer:
 - HFActionPolicy exact NF4 training-matched inference from existing code
 - preserves ALL role-authorized native observation text
 - replaces option-position dependence by canonical native action descriptions
 - only model choices control downstream actions; no native oracle teacher
 - SFT/DPO point to existing training entrypoints, not rewritten algorithms
The data AI/native verifier supplies truth, signed chance and business outcomes.
"""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from typing import Mapping,Sequence
import json,random,re
SRC=Path(__file__).resolve().parents[1]
SFT_ENTRY=SRC/"tools/kci_shared_semantic_train.py"
DPO_ENTRY=SRC/"tools/kci_shared_semantic_dpo.py"
HF_POLICY=SRC/"code/kci/agent_rl/hf_policy.py"
CCE_CORE=SRC/"code/kci/continuation.py"
DPR_CORE=SRC/"code/kci/repair.py"
# The trained Phase2 model uses ACCEPT/REJECT semantic candidate comparisons.
# Native KC8 menus carry action description in original role-visible source.
MENU_MARKER="Choose one CURRENTLY EXECUTABLE operation:"
MENU_LINE=re.compile(r"^([A-Z])\. (.*)$")
def file_sha(p):
    h=sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def canonical_actor_candidates(model_view:Mapping)->tuple[str,dict[str,str],str]:
    """Do NOT use data-AI evaluator-only labels or hidden world state."""
    if set(model_view)-{"actor","stage","model_prompt","prompt_sha256","available_actions",
                        "complete_role_observable_empirical_probability",
                        "no_ground_truth_label_or_specific_future_realization",
                        "no_hidden_teacher_or_future_truth",
                        "role_observation_sha256",
                        "actual_public_routing_and_automatic_source_contract_present",
                        "actual_signed_tool_status_is_visible"}:
        raise ValueError("unexpected model view field: do not pass evaluation metadata")
    source=model_view["model_prompt"]
    assert sha256(source.encode("utf8")).hexdigest()==model_view["prompt_sha256"]
    head,tag,tail=source.partition(MENU_MARKER)
    if not tag:raise ValueError("missing model-authorized menu boundary")
    ops=tuple(model_view["available_actions"])
    if not ops or len(ops)!=len(set(ops)) or "ABORT" not in ops:
        raise ValueError("must preserve all executable options including ABORT")
    descriptions={}
    letters=[]
    for line in tail.strip().splitlines():
        match=MENU_LINE.match(line.strip())
        if match:
            letters.append((match.group(1),match.group(2)))
    if len(letters)!=len(ops):
        raise ValueError(("option count mismatch",len(letters),len(ops)))
    # The native renderer binds ordered descriptions to same ordered IDs;
    # exact semantics, not alphabet position, are now scored.
    for act,(_,desc) in zip(ops,letters):
        assert desc and act not in descriptions
        descriptions[act]=desc
    return tuple(sorted(ops)),descriptions,head.rstrip()
def semantic_candidate_prompt(view:Mapping,action:str):
    actions,descriptions,head=canonical_actor_candidates(view)
    if action not in actions:raise ValueError("not an executable native action")
    native_actions="\n".join("- "+act+": "+descriptions[act] for act in actions)
    return (
        head+"\nAvailable executable native actions (canonical, no menu-letter position):\n"+
        native_actions+"\n\nCandidate action under consideration: "+action+
        "\nCandidate meaning: "+descriptions[action]+
        "\nEvaluate the candidate from the actor's actual delivered history, "
        "signed tool feedback, PUBLIC source uncertainty, declared recipient-and-"
        "purpose CI rules, legitimate business obligations and deadlines. "
        "Future signed events have not happened yet. No model opinion is a verifier."
        "\nAnswer A if this should be the preferred action, or B otherwise. "
        "Respond with exactly one letter.\nAnswer:"
    )
class FrozenSharedSemantic4B:
    def __init__(self,base:str,adapter:str,max_prompt_tokens:int=8192,
                 inference_precision:str="training"):
        from kci.agent_rl.hf_policy import HFActionPolicy
        self.hf=HFActionPolicy(model_path=base,adapter=adapter,trainable=False,
            quantization="4bit",dtype="bf16",max_prompt_tokens=max_prompt_tokens,
            allow_download=False,kbit_inference_precision=inference_precision,
            temperature=1.,explore_epsilon=0.)
        self.hf.model.eval()
        self.adapter_sha256=file_sha(Path(adapter)/"adapter_model.safetensors")
        self.cache={}
        self.forward_calls=0
        self.token_count=0
        self.cache_hits=0
        self.observation_count=0
        self.native_action_scoring="semantic_ACCEPT_REJECT"
    def distribution(self,view:Mapping):
        import torch
        actions,descriptions,head=canonical_actor_candidates(view)
        key=sha256(json.dumps([view["prompt_sha256"],actions,descriptions],
                              sort_keys=True).encode()).hexdigest()
        self.observation_count+=1
        if key in self.cache:
            self.cache_hits+=1
            return self.cache[key]
        values=[]
        with torch.inference_mode():
            for a in actions:
                text=semantic_candidate_prompt(view,a)
                tokenized=self.hf._input_ids(text)
                self.token_count+=int(tokenized.numel())
                logits=self.hf.log_distribution(text,("ACCEPT","REJECT"))
                values.append(float(logits[0]-logits[1]))
                self.forward_calls+=1
        probabilities=torch.softmax(torch.tensor(values),dim=0).tolist()
        if abs(sum(probabilities)-1)>1e-5 or any(p<0 for p in probabilities):
            raise AssertionError("invalid model softmax")
        result={a:float(p) for a,p in zip(actions,probabilities)}
        self.cache[key]=result
        return result
    def sample(self,view,rng:random.Random):
        dist=self.distribution(view)
        x=rng.random()
        cum=0.
        for a,p in dist.items():
            cum+=p
            if x<cum:
                return a,dist
        return list(dist)[-1],dist
    def costs(self):
        return {"candidate_model_forwards":self.forward_calls,
                "input_tokens":self.token_count,
                "cache_hits":self.cache_hits,
                "observations_scored":self.observation_count,
                "unique_action_distributions":len(self.cache)}
def current_model_entrypoints():
    refs={"SFT":SFT_ENTRY,"same_state_DPO":DPO_ENTRY,
          "HFActionPolicy":HF_POLICY,
          "CCE_continuation":CCE_CORE,
          "DPR_repair":DPR_CORE}
    if not all(p.is_file() for p in refs.values()):
        raise RuntimeError("A previously existing component is missing")
    return {k:{"path":str(p),"SHA256":file_sha(p)} for k,p in refs.items()}
