"""Exercise the actual shared loss and rule out future-label leakage at its readout."""
from learned_evidence import *

def run():
    while not (RUN/'structural_cases/RESULTS.json').exists():time.sleep(2)
    tokenizer,model=setup(17);model.eval();pool=palette(tokenizer)['train'];buckets=paired_states();checks=[]
    for role in ('Planner','Source','Worker'):
        pair=buckets[next(k for k in sorted(buckets) if k[0]==role)][0];start=time.perf_counter()
        q,a,tokens,estimates=joint_loss(tokenizer,model,pair,pool,317,0,True);assert torch.isfinite(q+a);(q+a).backward()
        assert any(p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum()>0) for p in model.parameters() if p.requires_grad)
        for row,pred in zip(pair,estimates):
            text,actions=actor_text(row['observation'],pool,317,pred)
            assert all(chr(65+i)+'='+pred[action] in text for i,action in enumerate(actions))
        model.zero_grad();checks.append({'role':role,'finite_nonzero_gradient':True,'seconds':time.perf_counter()-start});assert checks[-1]['seconds']<9.5
    row=pair[0];text=query(tokenizer,row,row['candidates'][0]['action'],pool,317)[0]
    with torch.inference_mode():logits,_,_=forward_targets(tokenizer,model,[text,text],['ANSWER:\n0','ANSWER:\n1'])
    difference=float((logits[0,-2]-logits[1,-2]).abs().max());assert difference<1e-5
    result={'shared_loss_role_checks':checks,'different_future_labels_decision_logit_max_difference':difference,'actor_estimates_match_action_order':True,'training_worker_source_sha256':hashlib.sha256((RUN/'learned_evidence.py').read_bytes()).hexdigest(),'errors':0}
    (RUN/'LEARNED_EVIDENCE_SMOKE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
