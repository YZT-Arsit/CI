"""Compare short, identical greedy prefixes across batch shapes and SDPA kernels."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    src = json.loads(Path(args.source).read_bytes())
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"":"cuda:0"}).eval()
    audit = []
    for name, backends in [
        ("auto", None),
        ("math", [SDPBackend.MATH]),
    ]:
        for inds in ([1], [0, 1, 2, 3], list(range(24))):
            x = tok([src[i]["prompt"] for i in inds], padding=True, return_tensors="pt", add_special_tokens=False).to(m.device)
            from contextlib import nullcontext
            with (sdpa_kernel(backends) if backends else nullcontext()), torch.inference_mode():
                out = m.generate(**x, max_new_tokens=10, do_sample=False, return_dict_in_generate=True, output_scores=True)
            idx = inds.index(1)
            ids = out.sequences[idx, x.input_ids.shape[1]:].tolist()
            best = [torch.topk(s[idx], 3) for s in out.scores[:3]]
            log = dict(kernel=name,batch=len(inds),padding_side=tok.padding_side,ids=ids,text=tok.decode(ids),
                       top=[list(zip(v.indices.tolist(),v.values.tolist())) for v in best])
            print(log, flush=True)
            audit.append(log)
    Path(args.out).write_text(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
