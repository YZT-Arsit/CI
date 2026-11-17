"""Exercise the actual loss functions with deterministic logits, without model load."""
import ast,json,random,sys
from pathlib import Path
R=Path(__file__).resolve().parent
env=json.loads((R.parent/'20261116_relation_solution/environment.json').read_text())
sys.path[:0]=[env['lm_dependency_path'],env['torch_path']]
import torch
torch.set_num_threads(1)

class Tokenizer:
    def encode(self,s,**kwargs):return [int(s)+1]

def function(path,name):
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={'torch':torch,'random':random,'answer_ids':lambda t:torch.tensor([4,5,6]),'query':lambda *a:('query',None),'actor_text':lambda obs,*a:('actor',obs['actions']),'response':lambda row,*a:('answer',row['optimal_indices'])}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace,name

def run(namespace,name,pair,weight=None):
    torch.manual_seed(19)
    q=torch.randn(sum(len(r['candidates']) for r in pair),4,10,requires_grad=True)
    a=torch.randn(2,4,10,requires_grad=True)
    def forward(tokenizer,model,texts,answers):
        logits=q if texts[0]=='query' else a
        ce=torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.ones(logits.shape[0]*4,dtype=torch.long),reduction='none').reshape(logits.shape[:2])
        return logits,ce,20
    namespace['forward_targets']=forward
    args=[Tokenizer(),None,pair,None,0,0]
    values=namespace[name](*args,True if weight is None else weight)
    loss=values[0]+values[1] if weight is None else values[3]
    grads=torch.autograd.grad(loss,[q,a],retain_graph=True)
    return values,grads,a

def main():
    old,name=function(R/'coverage_repair.py','loss_for');new,newname=function(R/'risk_repair.py','risk_loss')
    pair=[{'observation':{'actions':['x','y','z']},'candidates':[{'action':a,'safe_completion':int(i==0),'immediate_safe':i!=1} for i,a in enumerate(['x','y','z'])],'optimal_indices':[0]} for _ in range(2)]
    baseline,bg,_=run(old,name,pair);zero,zg,_=run(new,newname,pair,0.)
    assert all(torch.equal(x,y) for x,y in zip(baseline[:2],zero[:2]))
    assert all(torch.equal(x,y) for x,y in zip(bg,zg))
    active,ag,a=run(new,newname,pair,1.)
    assert active[2]>0 and all(torch.isfinite(g).all() for g in ag)
    safety_gradient=torch.autograd.grad(active[2],a)[0][:,-2]
    assert torch.all(safety_gradient[:,5]>0) and torch.all(safety_gradient[:,[4,6]]<0)
    assert torch.count_nonzero(safety_gradient[:,[0,1,2,3,7,8,9]])==0
    for row in pair:
        for c in row['candidates']:c['immediate_safe']=True
    tied,_,a=run(new,newname,pair,1.)
    assert tied[2].item()==0 and torch.count_nonzero(torch.autograd.grad(tied[2],a)[0])==0
    result={'lambda_zero_loss_and_gradients_exactly_equal':True,'safety_gradient_reduces_unsafe_and_increases_safe_legal_logits':True,'nonlegal_logits_have_zero_safety_gradient':True,'all_safe_loss_and_gradient_zero':True,'all_total_gradients_finite':True,'scope':'deterministic logits unit smoke of extracted actual functions; no model optimizer update'}
    (R/'RISK_LOSS_SMOKE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
