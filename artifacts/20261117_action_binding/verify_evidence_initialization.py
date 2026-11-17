"""Check all actual-initialization gradients; supplement the base-adapter smoke."""
from learned_evidence import *

def run():
    tokenizer,model=setup(17);folder=RUN/'query_pilot/independent';seal=json.loads((folder/'SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.train();pool=palette(tokenizer)['train'];buckets=paired_states();rows=[]
    for role in ('Planner','Source','Worker'):
        start=time.perf_counter();pair=buckets[next(k for k in sorted(buckets) if k[0]==role)][0];q,a,_,_=joint_loss(tokenizer,model,pair,pool,317,0,True);assert bool(torch.isfinite(q+a));(q+a).backward();grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and any(bool(g.abs().sum()>0) for g in grads)
        model.zero_grad();seconds=time.perf_counter()-start;assert seconds<9.5;rows.append({'role':role,'gradient_tensors':len(grads),'all_finite':True,'any_nonzero':True,'seconds':seconds})
    result={'adapter_hashes':seal['adapter_hashes'],'roles':rows,'source_sha256':hashlib.sha256((RUN/'learned_evidence.py').read_bytes()).hexdigest(),'errors':0};(RUN/'LEARNED_EVIDENCE_INITIALIZATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
