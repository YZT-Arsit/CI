"""Compare cached decoding with direct teacher-forcing for an identical prefix."""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = "Qwen3-4B-Instruct-2507"
    tok = AutoTokenizer.from_pretrained(p)
    src = json.loads(Path("diagnostics/initial_batch24/seed-42.json").read_bytes())
    txt = src[1]["prompt"]
    short = tok.apply_chat_template([{"role":"user","content":"What is 2+2? Answer with a number."}],tokenize=False,add_generation_prompt=True)
    m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.bfloat16).eval()
    logs = []
    for device in ("cpu","cuda"):
        m.to(device)
        for prompt in (txt,short):
            x = tok(prompt, add_special_tokens=False,return_tensors="pt").to(device)
            with torch.inference_mode():
                rows = {}
                for cache in (False,True):
                    result = m.generate(**x, max_new_tokens=6, do_sample=False, use_cache=cache)
                    ids=result[0,x.input_ids.shape[1]:].tolist()
                    rows[str(cache)] = [ids,tok.decode(ids)]
                xx = torch.cat([x.input_ids,torch.tensor([[27]],device=device)],dim=1)
                v=m(input_ids=xx,use_cache=False,logits_to_keep=1).logits[0,-1].float().topk(5)
                rows["direct"]=list(zip(v.indices.tolist(),v.values.tolist()))
            logs.append(dict(device=device,task="ci" if prompt==txt else "arithmetic",output=rows))
            print(logs[-1],flush=True)
    Path("diagnostics/prefix_comparison.json").write_text(json.dumps(logs,indent=2))


if __name__=="__main__":
    main()
