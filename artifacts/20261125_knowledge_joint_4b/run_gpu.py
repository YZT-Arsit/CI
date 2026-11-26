"""Shared Qwen3-4B menu policy, complete native graphs and exact end-to-end autograd."""
import gc, hashlib, json, random, sys, time
from collections import Counter
from pathlib import Path
import torch
from torch.utils.checkpoint import checkpoint
from objective import HERE, RHO, LAMBDA, compute, dump, read, scalars, tags

OLD=Path("/root/autodl-tmp/kci_v9_training_18828")
sys.path.insert(0,str(OLD))
from v9_experiment_policy import Policy,digest
BASE=Path("/root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms")
START=OLD/"proof_cce_7/adapter"
OLD_FREEZE=Path("/root/autodl-tmp/kci_continuation_4b_18828/FREEZE.json")
SEED=7
def h(x):
    return hashlib.sha256(x.encode()).hexdigest()

class Menus:
    def __init__(self,pol,obs):
        self.pol=pol;self.rows={}
        for t,r in obs.items():
            actions=r["actions"]
            ordered=sorted(actions,key=lambda a:h("7|"+a))
            stem="Use only your role-visible information. Select a task-sufficient and contextually appropriate executable action. Reply with its one-character ID."
            text=stem+"\n"+json.dumps(r["context"],ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\nExecutable actions:\n"+"\n".join(pol.labels[i]+" = "+a for i,a in enumerate(ordered))
            ids=pol.tok.apply_chat_template([dict(role="user",content=text)],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            codes=[pol.codes[ordered.index(a)] for a in actions]
            assert len(set(codes))==len(actions) and len(ids)<pol.net.config.max_position_embeddings
            self.rows[t]=(ids,codes)
    def batches(self,keys):
        batch=[]
        for t in sorted(set(keys),key=lambda t:(len(self.rows[t][0]),t)):
            if batch and (len(batch)==3 or len(self.rows[t][0])*(1+len(batch))>3600):
                yield batch;batch=[]
            batch.append(t)
        if batch:yield batch
    def one(self,batch,cost,grad=False,generate=False):
        inputs=[self.rows[t][0] for t in batch];codes=[self.rows[t][1] for t in batch]
        w=max(map(len,inputs));pol=self.pol
        x=torch.tensor([[pol.tok.pad_token_id]*(w-len(i))+i for i in inputs],device="cuda")
        mask=torch.tensor([[0]*(w-len(i))+[1]*len(i) for i in inputs],device="cuda")
        cost.update(batches=1,queries=len(batch),tokens=sum(map(len,inputs)),padded_tokens=x.numel())
        if generate:
            gen=pol.net.generate(input_ids=x,attention_mask=mask,max_new_tokens=1,do_sample=False,
                    use_cache=False,top_k=None,top_p=None,temperature=None,pad_token_id=pol.tok.pad_token_id,
                    prefix_allowed_tokens_fn=lambda i,_:codes[i],return_dict_in_generate=True,output_scores=True)
            z=gen.scores[0].float();chosen=gen.sequences[:,-1].tolist()
            cost["generated_ids"]+=len(batch)
            for i,c in enumerate(codes):assert chosen[i]==c[z[i,c].argmax().item()] or z[i,chosen[i]]==z[i,c].max()
            picked={t:codes[i].index(chosen[i]) for i,t in enumerate(batch)}
        else:
            def forward(xx,mm):
                return pol.net(input_ids=xx,attention_mask=mm,position_ids=(mm.cumsum(1)-1).clamp_min(0),use_cache=False,logits_to_keep=1).logits[:,-1].float()
            z=checkpoint(forward,x,mask,use_reentrant=False) if grad else forward(x,mask)
            if grad:cost.update(recomputed_batches=1,recomputed_tokens=sum(map(len,inputs)),recomputed_padded_tokens=x.numel())
            picked={}
        return {t:z[i,c] for i,(t,c) in enumerate(zip(batch,codes))},picked

def train_order(groups,epoch):
    seq=sorted((g for g in groups if g["split"]=="train"),key=lambda g:h(f"{SEED}|{epoch}|{g['root']}"))
    return seq

def cost_plan(q,groups):
    c=Counter()
    for g in train_order(groups,1):
        for batch in q.batches(tags(g["worlds"]["original"])):
            w=max(len(q.rows[t][0]) for t in batch)
            c.update(batches=2,queries=2*len(batch),tokens=2*sum(len(q.rows[t][0]) for t in batch),padded_tokens=2*w*len(batch))
    return dict(c)

def calibration(pol,q,groups,obs):
    counts=Counter()
    group=train_order(groups,1)[0];w=group["worlds"]["original"];keys=tags(w)
    batch=max(q.batches(keys),key=lambda b:len(b)*max(len(q.rows[t][0]) for t in b))
    torch.cuda.synchronize();tic=time.monotonic()
    with torch.no_grad():q.one(batch,counts)
    torch.cuda.synchronize();dt=time.monotonic()-tic
    z={}
    for b in q.batches(keys):z.update(q.one(b,Counter(),grad=True)[0])
    loss,_=compute(group,obs,z,"B")
    slopes=torch.autograd.grad(loss,list(z.values()),retain_graph=True)
    param=next(p for n,p in pol.net.named_parameters() if "v_proj.lora_B" in n and p.requires_grad)
    def role_vjp(up):
        items=[(value,v) for (k,value),v in zip(z.items(),slopes) if (k==w["head"])==up]
        return torch.autograd.grad([v for v,_ in items],param,[s for _,s in items],retain_graph=True)[0]
    a,b=role_vjp(True),role_vjp(False)
    full=torch.autograd.grad(loss,param)[0]
    err=(full-a-b).norm()/full.norm()
    assert a.norm()>0 and b.norm()>0 and err<0.02
    expected=cost_plan(q,groups)
    cap=180+max(3.,dt/counts["padded_tokens"]*expected["padded_tokens"])*12
    dump(HERE/"BUDGET.json",dict(calibration=counts,forward_seconds=dt,expected_per_arm=expected,training_seconds_ceiling_per_arm=cap,
        recomputation="One extra checkpoint forward per training batch",real_4b_vjp=dict(root=group["root"],parameter="first v_proj.lora_B",up=a.norm().item(),down=b.norm().item(),relative_sum_error=err.item()),
        resources=dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__),rho=RHO,history_weight=LAMBDA,epochs=2,lr=2e-5,seed=SEED))

def evaluate(pol,q,groups,obs,name):
    cnt=Counter();tic=time.monotonic();z={};pick={}
    keys={t for g in groups for w in g["worlds"].values() for t in tags(w)}
    pol.net.eval()
    with torch.inference_mode():
        for batch in q.batches(keys):
            zz,ii=q.one(batch,cnt,generate=True);z.update({t:v.double().cpu() for t,v in zz.items()});pick.update(ii)
    rows=[]
    for g in groups:
        r={k:g[k] for k in ("domain","source","root","row","split")}
        for mode in ("soft","deployed"):
            # Turn the ID actually emitted by generate into a deterministic policy, including tie handling.
            policy_z=z if mode=="soft" else {t:torch.nn.functional.one_hot(torch.tensor(i),len(z[t])).double() for t,i in pick.items()}
            _,m=compute(g,obs,policy_z,greedy=mode=="deployed")
            if "authorized" in g["worlds"]:
                _,am=compute(g,obs,policy_z,greedy=mode=="deployed",norm="authorized")
                m["restoration"]=am["restoration"]
            r[mode]=scalars(m)
        rows.append(r)
    torch.cuda.synchronize()
    dump(HERE/(name+".json"),dict(rows=rows,menu_scores={t:v.tolist() for t,v in z.items()},selected=pick,cost=dict(cnt,seconds=time.monotonic()-tic)))

def train(arm,pol,q,groups,obs,budget):
    params=[p for p in pol.net.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=2e-5,weight_decay=0.)
    cnt=Counter();elapsed=0.;rows=[]
    pol.net.eval()  # deterministic current policy, gradients enabled; LoRA dropout is zero.
    for epoch in (1,2):
        tic=time.monotonic()
        for g in train_order(groups,epoch):
            if elapsed+time.monotonic()-tic>budget["training_seconds_ceiling_per_arm"]:raise RuntimeError("time budget exhausted")
            opt.zero_grad(set_to_none=True);z={}
            for batch in q.batches(tags(g["worlds"]["original"])):
                z.update(q.one(batch,cnt,grad=True)[0])
            loss,m=compute(g,obs,z,arm)
            loss.backward();cnt["backward_roots"]+=1
            gn=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            rows.append(dict(epoch=epoch,root=g["root"],domain=g["domain"],step=len(rows)+1,loss=loss.item(),grad=gn.item(),**scalars(m)))
        torch.cuda.synchronize();elapsed+=time.monotonic()-tic
        dump(HERE/(arm+"_updates.json"),rows)
        evaluate(pol,q,groups,obs,arm+str(epoch))
        print(arm,epoch,"completed",round(elapsed,2),flush=True)
    assert all(cnt[k]==v for k,v in budget["expected_per_arm"].items())
    assert elapsed<=budget["training_seconds_ceiling_per_arm"]
    pol.net.save_pretrained(HERE/(arm+"_adapter"))
    dump(HERE/(arm+"_cost.json"),dict(**cnt,seconds=elapsed,peak_bytes=torch.cuda.max_memory_allocated(),adapter=digest(HERE/(arm+"_adapter/adapter_model.safetensors"))))

def main():
    assert json.loads((HERE/"CPU_CHECK.json").read_text())["verdict"]=="PASS"
    freeze=json.loads((HERE/"FREEZE.json").read_text())
    for k,v in freeze["files"].items():assert digest(HERE/k)==v,k
    old=json.loads(OLD_FREEZE.read_text())
    assert all(digest(BASE/k)==v for k,v in old["base"].items())
    assert digest(START/"adapter_model.safetensors")==old["start"]==freeze["start"]
    groups,obs=read("groups"),read("observations")
    assert all(g["matches"] for g in groups) # equal group weighting of the history and policy objectives
    for arm in "ABC":
        if (HERE/(arm+"_cost.json")).exists():continue
        torch.manual_seed(SEED);random.seed(SEED);torch.cuda.reset_peak_memory_stats()
        pol=Policy(BASE,START,trainable=True);pol.net.gradient_checkpointing_disable()
        q=Menus(pol,obs)
        if not (HERE/"BUDGET.json").exists():calibration(pol,q,groups,obs)
        if not (HERE/"initial.json").exists():evaluate(pol,q,groups,obs,"initial")
        train(arm,pol,q,groups,obs,json.loads((HERE/"BUDGET.json").read_text()))
        del pol,q;gc.collect();torch.cuda.empty_cache()
    print("COMPLETE",flush=True)
if __name__=="__main__":main()
