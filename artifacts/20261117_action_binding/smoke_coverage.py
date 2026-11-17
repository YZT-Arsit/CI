"""Check restored tied-state coverage and equivalence of the unweighted loss."""
from coverage_repair import *
from audit_data import check

def run():
    buckets=full_buckets();rows=[r for pairs in buckets.values() for pair in pairs for r in pair];assert not [e for r in rows for e in check(r)];tied=[r for r in rows if len(r['optimal_indices'])==len(r['candidates'])];assert tied
    assert any(r['observation']['role']=='Planner' and r['observation']['contract']['permission'] and all(c['safe_completion'] for c in r['candidates']) for r in tied)
    tokenizer,model=setup(17);folder=RUN/'learned_evidence/evidence';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['train'];pair=next(pair for key,pairs in sorted(buckets.items()) if key[0]=='Worker' for pair in pairs if 0<sum(c['safe_completion'] for r in pair for c in r['candidates'])<sum(len(r['candidates']) for r in pair))
    with torch.inference_mode():
        old=joint_loss(tokenizer,model,pair,pool,731,0,True);new=loss_for(tokenizer,model,pair,pool,731,0,False)
    difference=max(float((old[i]-new[i]).abs()) for i in (0,1));assert difference<1e-6;assert old[2:4]==new[2:4]
    start=time.perf_counter();q,a,_,_,counts=loss_for(tokenizer,model,pair,pool,731,0,True);(q+a).backward();grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None];assert grads and all(bool(torch.isfinite(g).all()) for g in grads) and any(bool(g.abs().sum()>0) for g in grads);seconds=time.perf_counter()-start;assert seconds<9.5
    result={'independent_label_states':len(rows),'retained_all_tied_states':len(tied),'unweighted_loss_difference':difference,'all_gradient_tensors_finite':len(grads),'balanced_batch_query_labels':counts,'balanced_backward_seconds':seconds,'source_sha256':hashlib.sha256((RUN/'coverage_repair.py').read_bytes()).hexdigest(),'errors':0};(RUN/'COVERAGE_SMOKE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
