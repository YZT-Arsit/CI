"""Check actual initialization gradients and identity-view consistency; no updates."""
from repair import *

def main():
    folder=HERE.parent/'20261117_action_binding/coverage_repair/class_balanced';tokenizer,model=setup(17);set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));model.train();pool=palette(tokenizer)['train'];buckets=full_buckets();records=[]
    for role in ['Planner','Source','Worker']:
        key=next(k for k in sorted(buckets) if k[0]==role);pair=buckets[key][0];seeds={k:27117 for k in ['names','facts','plans','options','links']};start=time.perf_counter();loss,parts,tokens,estimates,eligible=loss_for_views(tokenizer,model,pair,pool,[seeds,seeds],0,1.);assert abs(float(parts['query_js'].detach()))<1e-7 and abs(float(parts['actor_js'].detach()))<1e-7 and eligible==2;model.zero_grad();loss.backward();grads=[p.grad for p in model.parameters() if p.requires_grad];assert all(g is not None and torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g)>0 for g in grads);records.append({'role':role,'gradient_tensors':len(grads),'all_finite':True,'nonzero_gradients':True,'identity_query_and_actor_js_zero':True,'seconds':time.perf_counter()-start,'input_tokens':tokens})
    result={'all_roles_passed':True,'optimizer_updates':0,'roles':records};(HERE/'REPAIR_MODEL_SMOKE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
