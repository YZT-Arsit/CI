"""Role-routed pretrained HF/PEFT action policy for KCI A -> R -> B -> C.

Load ONE frozen underlying causal LM. Route lightweight adapters by the
actually acting role and native decision stage; do not mix SourceAgent local
repair gradients into Coordinator or upstream RecordsAgent.
Caller controls the environment; only actor-visible observations enter LLM.

This is the actual GPU serving adapter interface for future role LoRA runs,
not a substitute for 7B weight training on the CPU Ollama GGUF checkpoint.
"""
from __future__ import annotations
import random
from pathlib import Path
import torch
from kci_llm_text_surface import render_role_text

def learning_role(observation):
    stage=observation.get("stage",observation.get("current_stage"))
    if stage=="A_SEND":return "RecordsAgent"
    if stage=="preflight":return "Coordinator"
    if stage=="release":return "SourceAgent"
    if stage=="choose_fact":return "BusinessExecutor"
    return "None"

class RoleRoutedHFCategoricalPolicy:
    def __init__(self, *, model_path, adapters, quantization="4bit",
                 dtype="bf16", max_prompt_tokens=4096):
        """adapters: role-name -> directory with trained PEFT adapter."""
        if not adapters:raise ValueError("Provide at least one trained role adapter")
        allowed={"RecordsAgent","Coordinator","SourceAgent","BusinessExecutor"}
        if not set(adapters).issubset(allowed):raise ValueError("Unknown trained role")
        from kci.agent_rl.hf_policy import HFActionPolicy
        first_role=next(iter(adapters))
        self.base=HFActionPolicy(
            model_path=model_path,adapter=str(Path(adapters[first_role])),
            trainable=False,quantization=quantization,dtype=dtype,
            max_prompt_tokens=max_prompt_tokens,
            allow_download=False,temperature=1.0,explore_epsilon=0.0)
        self.role_map={first_role:"default"}
        # Under PEFT, the first adapter supplied to from_pretrained is
        # normally called "default"; additional adapters are loaded by name.
        for role,folder in adapters.items():
            if role==first_role:continue
            self.base.model.load_adapter(str(Path(folder)),adapter_name=role,
                                         is_trainable=False)
            self.role_map[role]=role
        self.base.model.requires_grad_(False)
        self.base.model.eval()
        self.model_id=f"role-peft:{model_path}:"+",".join(sorted(self.role_map))

    def log_distribution(self,observation):
        if observation.get("done"):raise ValueError("Cannot infer on terminal state")
        options=tuple(observation.get("legal_actions",()))
        if not options:raise ValueError("Missing authorized native options")
        prompt=render_role_text(observation)
        role=learning_role(observation)
        if role in self.role_map:
            self.base.model.set_adapter(self.role_map[role])
            with torch.no_grad():
                return self.base.log_distribution(prompt,options)
        # Roles that have not been trained use the original frozen base model.
        with self.base.model.disable_adapter(),torch.no_grad():
            return self.base.log_distribution(prompt,options)

    def choose(self,observation,rng:random.Random,greedy=False):
        actions=tuple(observation["legal_actions"])
        with torch.no_grad():
            logp=self.log_distribution(observation)
            probs=logp.exp().detach().cpu().tolist()
        if greedy:
            idx=max(range(len(probs)),key=lambda j:probs[j])
        else:
            u=rng.random()
            cumulative=0.;idx=len(probs)-1
            for j,p in enumerate(probs):
                cumulative+=p
                if u<=cumulative:idx=j;break
        return actions[idx],float(logp[idx]),learning_role(observation)
