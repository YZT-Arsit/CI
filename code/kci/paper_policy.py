"""One shared HF/LoRA policy. Both rollout and DPO use the same native-action distribution."""
from collections import defaultdict
import math
from pathlib import Path
import random
from .paper_alignment import prompt
from .paper_worlds import canon, filehash, key


class SharedLM:
    def __init__(self,base,adapter):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        self.torch=torch
        self.tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        dtype=torch.bfloat16 if self.device=="cuda" else torch.float32
        net=AutoModelForCausalLM.from_pretrained(base,dtype=dtype,local_files_only=True,attn_implementation="sdpa")
        self.net=PeftModel.from_pretrained(net,adapter,is_trainable=True,local_files_only=True).to(self.device)
        self.net.eval()
        self.net.config.use_cache=False
        self.version=filehash(Path(adapter)/"adapter_model.safetensors")
        self.labels=[]
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            ids=self.tok.encode(c,add_special_tokens=False)
            if len(ids)!=1 or self.tok.decode(ids)!=c:raise ValueError("Menu tokens are ambiguous")
            self.labels.append(ids[0])
        self.cache={}

    def logits(self,visible,actions):
        menu="\n".join(chr(65+i)+". "+canon(a) for i,a in enumerate(actions))
        if not 0<len(actions)<=26:raise ValueError("Unsupported action menu")
        message=("Choose the next executable action for your role. Obey the stated recipient and purpose "
                 "rules, preserve safe task completion, then minimize successful execution cost. "
                 "Use your authorized observations and delivered history. Answer with one option letter.\n"
                 +visible+"\n"+menu)
        ids=self.tok.apply_chat_template([{"role":"user","content":message}],tokenize=True,add_generation_prompt=True,
                                         enable_thinking=False,return_dict=False)
        if len(ids)>self.net.config.max_position_embeddings:raise ValueError("Observation exceeds model context")
        x=self.torch.tensor([ids],device=self.device)
        return self.net(input_ids=x,logits_to_keep=1,use_cache=False).logits[0,-1,self.labels[:len(actions)]].float()

    def choose(self,ob):
        actions=sorted(ob["legal_actions"],key=canon)
        if len(actions)==1:return actions[0]
        p=prompt(ob)
        if p not in self.cache:
            with self.torch.no_grad():self.cache[p]=actions[int(self.logits(p,actions).argmax())]
        return self.cache[p]

    def fit_schedule(self,steps,out,lr=2e-5,beta=.35):
        """One arithmetic mean over the four sealed exposure slots in each update."""
        import json
        torch=self.torch
        frozen={}
        for batch in steps:
            if len(batch)!=4:raise ValueError("Unequal exposure budget")
            for r in batch:
                p=r["prompt"]
                actions=sorted(json.loads(p)["legal_actions"],key=canon)
                if r["chosen"]==r["rejected"] or not all(r[k] in actions for k in ("chosen","rejected")):
                    raise ValueError("Invalid native contrast")
                if r["objective"] not in ("ce","dpo"):raise ValueError("Unknown objective")
                if p not in frozen:
                    with torch.no_grad():frozen[p]=(actions,self.logits(p,actions).detach().cpu())
        params=[p for p in self.net.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=0)
        logs=[]
        for i,batch in enumerate(steps):
            opt.zero_grad(set_to_none=True)
            total=0.
            for r in batch:
                actions,ref=frozen[r["prompt"]]
                pos,neg=(actions.index(r[k]) for k in ("chosen","rejected"))
                v=self.logits(r["prompt"],actions)
                margin=(v[pos]-v[neg])-(ref[pos]-ref[neg]).to(self.device)
                if i==0 and abs(float(margin.detach()))>1e-4:raise RuntimeError("Frozen baseline mismatch")
                if r["objective"]=="ce":
                    loss=-torch.log_softmax(v,dim=0)[pos]
                else:
                    loss=-torch.nn.functional.logsigmoid(beta*margin)
                (loss/len(batch)).backward()
                total+=float(loss.detach())/len(batch)
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step()
            logs.append(dict(step=i+1,mean_loss=total,grad_norm=float(norm),row_ids=[key(r) for r in batch]))
        initial=self.version
        self.net.save_pretrained(out,safe_serialization=True)
        self.version=filehash(Path(out)/"adapter_model.safetensors")
        if self.version==initial:raise RuntimeError("No weight update")
        self.cache.clear()
        return dict(initial=initial,updated=self.version,lr=lr,beta=beta,normalizer=4,steps=logs)

    def train(self,rows,out,steps=16,lr=2e-5,beta=.35,seed=7,lambdas=None):
        """Eq. 14, one sample per present source per update, frozen pi0 = collection snapshot."""
        import json
        weights=dict(ci=1.,future=1.,repair=1.) if lambdas is None else lambdas
        if set(weights)!={"ci","future","repair"} or any(not math.isfinite(x) or x<0 for x in weights.values()):
            raise ValueError("Invalid source weights")
        if steps<=0 or lr<=0 or beta<=0:raise ValueError("Invalid training configuration")
        rng=random.Random(seed);groups=defaultdict(list);references={}
        for r in rows:
            if r["source"] not in weights or r["current_policy"]!=self.version or r["evidence"] not in (
                    "verified_violation_and_witness","exact_fixed_reference") or not 0<r["weight"]<=1:
                raise ValueError("Unverified or stale preference")
            if not weights[r["source"]]:continue
            actions=sorted(json.loads(r["prompt"])["legal_actions"],key=canon)
            pos,neg=actions.index(r["chosen"]),actions.index(r["rejected"])
            if pos==neg or key(r["prompt"])!=r["observation_id"]:raise ValueError("Mismatched pair")
            groups[r["source"]].append((r,actions,pos,neg))
        if not groups:raise ValueError("No verified supervision in the selected arms")
        for group in groups.values():
            for r,actions,_,_ in group:
                p=r["prompt"]
                if p not in references:
                    with self.torch.no_grad():references[p]=self.logits(p,actions).detach().cpu()
        params=[p for p in self.net.parameters() if p.requires_grad]
        opt=self.torch.optim.AdamW(params,lr=lr,weight_decay=0)
        logs=[]
        for step in range(steps):
            opt.zero_grad(set_to_none=True);loss_total=0.;ids=[]
            sources=sorted(groups)
            schedule=[sources[(step+i)%len(sources)] for i in range(3)] if len(sources)<3 else sources
            for source in schedule:
                items=groups[source]
                r,actions,pos,neg=rng.choice(items)
                # Softmax normalizers cancel for same-observation action log ratios.
                v=self.logits(r["prompt"],actions)
                ref=references[r["prompt"]]
                margin=(v[pos]-v[neg])-(ref[pos]-ref[neg]).to(self.device)
                if step==0 and margin.detach().abs().item()>1e-4:raise RuntimeError("pi0 path mismatch")
                loss=(-self.torch.nn.functional.logsigmoid(beta*margin)*r["weight"]*weights[source]
                      /schedule.count(source))
                loss.backward();loss_total+=float(loss.detach());ids.append(key(r))
            norm=self.torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step(); logs.append(dict(step=step+1,loss=loss_total,grad_norm=float(norm),row_ids=ids))
        self.net.save_pretrained(out,safe_serialization=True)
        old=self.version;self.version=filehash(Path(out)/"adapter_model.safetensors")
        self.cache.clear()
        if self.version==old:raise RuntimeError("Weights did not change")
        return dict(pi0=old,updated=self.version,steps=logs,source_lambda=weights,beta=beta,
                    admitted_counts={k:len(v) for k,v in groups.items()},seed=seed,lr=lr)

