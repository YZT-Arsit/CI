"""Actual-checkpoint nonidentity consistency gradients, bounded per role."""
from repair import *

def main():
    folder=HERE.parent/'20261117_action_binding/coverage_repair/class_balanced';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.train();pool=palette(tokenizer)['train'];buckets=full_buckets();records=[]
    for role in ['Planner','Source','Worker']:
        def timeout():
            (HERE/'VARIED_SMOKE_TIMEOUT.json').write_text(json.dumps({'role':role,'limit_seconds':9.5}));os._exit(124)
        key=next(k for k in sorted(buckets) if k[0]==role);pair=buckets[key][0];views=[{k:seed+i*39 for i,k in enumerate(['names','facts','plans','options','links'])} for seed in [27117,91722]];timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();loss,parts,tokens,estimates,eligible=loss_for_views(tokenizer,model,pair,pool,views,0,1.);consistency=parts['query_js']+parts['actor_js'];grads=torch.autograd.grad(consistency,[p for p in model.parameters() if p.requires_grad],allow_unused=True);assert torch.isfinite(loss) and all(g is not None and torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g)>0 for g in grads);timer.cancel();records.append({'role':role,'query_js':float(parts['query_js'].detach()),'actor_js':float(parts['actor_js'].detach()),'eligible_states':eligible,'all120_consistency_gradients_finite':len(grads)==120,'nonzero_consistency_gradient':True,'seconds':time.perf_counter()-start});del grads,loss,parts,consistency
    (HERE/'VARIED_MODEL_SMOKE.json').write_text(json.dumps({'roles':records,'optimizer_updates':0,'all_roles_passed':True,'adapter_hashes':seal['adapter_hashes']},indent=2));print(json.dumps(records))

if __name__=='__main__':main()
