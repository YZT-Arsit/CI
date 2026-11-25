"""Greedy native action choice by full semantic-completion likelihood, one shared frozen adapter."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def semantic_logps(net,tok,prompt,actions):
    """Return differentiable summed completion log probabilities in display order."""
    import torch
    prefix=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=True,
                 add_generation_prompt=True,enable_thinking=False)
    scores=[]
    for start in range(0,len(actions),4):
        suffix=[tok.encode(a,add_special_tokens=False)+[tok.eos_token_id] for a in actions[start:start+4]]
        width=max(map(len,suffix))
        if len(prefix)+width>net.config.max_position_embeddings:raise ValueError("Input exceeds model window")
        ids=torch.tensor([prefix+s+[tok.pad_token_id]*(width-len(s)) for s in suffix],device=net.device)
        mask=torch.tensor([[1]*(len(prefix)+len(s))+[0]*(width-len(s)) for s in suffix],device=net.device)
        lp=net(input_ids=ids,attention_mask=mask,use_cache=False,logits_to_keep=width+1).logits[:,:-1].float().log_softmax(-1)
        for i,s in enumerate(suffix):
            scores.append(lp[i,torch.arange(len(s),device=net.device),s].sum())
    return torch.stack(scores)


def run(base,adapter,inputs,out):
    import torch
    import transformers
    from peft import PeftModel
    tok=transformers.AutoTokenizer.from_pretrained(base,local_files_only=True)
    net=transformers.AutoModelForCausalLM.from_pretrained(base,dtype=torch.bfloat16,
              attn_implementation="sdpa",local_files_only=True)
    net=PeftModel.from_pretrained(net,adapter,is_trainable=False,local_files_only=True).to("cuda").eval()
    with gzip.open(inputs,"rt",encoding="utf8") as f:records=json.load(f)
    choices={}
    with torch.inference_mode():
        for r in records:
            scores=semantic_logps(net,tok,r["prompt"],r["actions"]).tolist()
            choices[r["id"]]=dict(action=r["actions"][max(range(len(scores)),key=scores.__getitem__)],logprob=scores)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    out.write_text(json.dumps(dict(input_sha256=digest(inputs),adapter_sha256=digest(adapter/"adapter_model.safetensors"),
        base_shards={p.name:digest(p) for p in sorted(base.glob("model-*.safetensors"))},
        torch=torch.__version__,transformers=transformers.__version__,
        decoding="greedy among semantic completions; summed assistant tokens plus EOS, prompt masked",
        choices=choices),ensure_ascii=False),encoding="utf8")


if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for n in ("base","adapter","inputs","output"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.base,a.adapter,a.inputs,a.output)

