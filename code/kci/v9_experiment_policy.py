"""Frozen V9 pilot scoring, optimizer and policy replay on a source-bound native graph."""
from pathlib import Path
from fractions import Fraction
from hashlib import sha256
from statistics import mean
import gzip
import json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel
from v9_policy import semantic_logps


def loadgz(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return json.load(f)


def dump(path,data):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")
    tmp.replace(path)


def digest(path):
    h=sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()


class Policy:
    def __init__(self,base,adapter,trainable=False):
        self.tok=AutoTokenizer.from_pretrained(base,local_files_only=True)
        net=AutoModelForCausalLM.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,attn_implementation="sdpa")
        self.net=PeftModel.from_pretrained(net,adapter,is_trainable=trainable,local_files_only=True).to("cuda").eval()
        self.net.config.use_cache=False
        if trainable:
            self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.version=digest(adapter/"adapter_model.safetensors")
        self.labels=[]
        self.codes=[]
        for label in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            tokens=self.tok.encode(label,add_special_tokens=False)
            if len(tokens)==1 and tokens[0] not in self.codes:
                self.labels.append(label)
                self.codes.append(tokens[0])

    def balanced(self,r,individual=False):
        actions=r["actions"];n=len(actions)
        if n>len(self.codes):raise ValueError("Action code is not single token")
        stem=r["prompt"].split("\nExecutable actions:\n")[0].replace(
            "Return one action exactly as shown, not its list position.","Reply with only the numeric ID of one action.")
        if n>10:stem=stem.replace("numeric ID","one-character ID")
        views=[]
        for inv in (False,True):
            ordered=actions[::-1] if inv else actions
            text=stem+"\nExecutable actions:\n"+"\n".join(f"{self.labels[j]} = {a}" for j,a in enumerate(ordered))
            ids=self.tok.apply_chat_template([{"role":"user","content":text}],tokenize=True,
                      add_generation_prompt=True,enable_thinking=False)
            x=torch.tensor([ids],device=self.net.device)
            if len(ids)>self.net.config.max_position_embeddings:raise ValueError("Context overflow")
            logits=self.net(input_ids=x,use_cache=False,logits_to_keep=1).logits[0,-1,self.codes[:n]].float()
            views.append(logits.flip(0) if inv else logits)
        return views if individual else torch.stack(views).mean(0).log_softmax(0)

    def score(self,r,domain):
        if domain=="neg":return self.balanced(r)
        return semantic_logps(self.net,self.tok,r["prompt"],r["actions"])


def evaluate(policy,gs,prompts):
    all_scores={}
    with torch.inference_mode():
        for g in gs:
            tags=[g["first_input"]]+[b["policy_input"] for bs in g["paths"].values() for b in bs]
            for t in tags:
                if t not in all_scores:
                    all_scores[t]=policy.score(prompts[t],g["domain"]).tolist()
    result=from_scores(gs,prompts,all_scores)
    return dict(weights=policy.version,root_results=result,scores=all_scores)


def from_scores(gs,prompts,scores):
    def pick(t):
        return prompts[t]["actions"][max(range(len(scores[t])),key=scores[t].__getitem__)]
    roots=[]
    for g in gs:
        q={};ref={};events={}
        for a,branches in g["paths"].items():
            selected=[]
            for b in branches:
                t=b["policy_input"];v=b["outcomes"][pick(t)]
                selected.append(dict(input=t,action=pick(t),mass=b["probability"],condition=b["condition"],
                    opportunity=any(z["stsr"] and not z["abort"] for z in b["outcomes"].values()),**v))
            q[a]={k:float(sum(Fraction(o["mass"])*o[k] for o in selected)) for k in
                  ("success","direct","inferred","stsr","abort","opportunity")}
            q[a]["retained_mass"]=float(sum(Fraction(o["mass"])*(o["opportunity"] and not
                    (o["direct"] or o["inferred"] or o["abort"])) for o in selected))
            events[a]=selected
            ref[a]=float(sum(Fraction(b["probability"])*b["reference_stsr"] for b in branches))
        assert ref==g["reference_stsr"] or all(abs(ref[k]-g["reference_stsr"][k])<1e-12 for k in ref)
        pos,neg=g["chosen"],g["rejected"];first=g["first_input"];actions=prompts[first]["actions"]
        upstream=scores[first][actions.index(pos)]-scores[first][actions.index(neg)]
        chosen=pick(first)
        # The positive reference branch gives every root at least one legal useful disclosure opportunity.
        retained=q[pos]["retained_mass"]/q[pos]["opportunity"]
        gap=q[pos]["stsr"]-q[neg]["stsr"]
        roots.append(dict(id=g["id"],root=g["source_root"],split_group=g["split_group"],domain=g["domain"],
            split=g["split"],weight_pair=[pos,neg],upstream_margin=upstream,first=chosen,
            qpi=q,qref=ref,qpi_gap=gap,rank=int(upstream>0),value_rank=int(gap>0),value_tie=int(gap==0),
            legal_disclosure_retention=retained,actual=q[chosen],events=events))
    return roots


def length_check(policy,gs,prompts):
    records=[]
    with torch.inference_mode():
        for g in gs:
            for a,bs in g["paths"].items():
                for b in bs:
                    t=b["policy_input"];r=prompts[t];acts=r["actions"]
                    raw=semantic_logps(policy.net,policy.tok,r["prompt"],acts)
                    lengths=[len(policy.tok.encode(x,add_special_tokens=False))+1 for x in acts]
                    per=raw/torch.tensor(lengths,device=raw.device)
                    u,v=policy.balanced(r,True)
                    choices={k:acts[int(x.argmax())] for k,x in
                             (("sum",raw),("mean",per),("equal",(u+v)/2),("forward",u),("reverse",v))}
                    records.append(dict(id=g["id"],t=t,first=a,choices=choices,lengths=lengths,
                                    actions=acts,sum_scores=raw.tolist(),mean_scores=per.tolist(),
                                    equal_scores=((u+v)/2).tolist(),lawful_available=any(x["stsr"] for x in b["outcomes"].values())))
    return records

