"""CPU loss checks for semantic alignment, OTHER and identity gradients."""
import json
from pathlib import Path
from two_view_loss import torch,semantic_log_probs,js
R=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);a=torch.tensor([2.,-1.,.5,3.,-.2],requires_grad=True);b=a.detach().clone().requires_grad_();value=js(semantic_log_probs(a,[0,3]),semantic_log_probs(b,[0,3]));ga,gb=torch.autograd.grad(value,[a,b]);assert value.item()==0 and torch.allclose(ga,torch.zeros_like(ga),atol=1e-7) and torch.allclose(gb,torch.zeros_like(gb),atol=1e-7)
    permuted=a.detach()[torch.tensor([3,1,2,0,4])].clone().requires_grad_();aligned=js(semantic_log_probs(a,[0,3]),semantic_log_probs(permuted,[3,0]));wrong=js(semantic_log_probs(a,[0,3]),semantic_log_probs(permuted,[0,3]));assert abs(aligned.item())<1e-7 and wrong.item()>0
    logp=semantic_log_probs(a,[0,3]);p=logp.exp();raw=torch.softmax(a,-1);assert torch.allclose(p,torch.stack([raw[0],raw[3],raw[1]+raw[2]+raw[4]])) and torch.allclose(p.sum(),torch.tensor(1.))
    left=torch.tensor([[4.,-2.,0.],[-2.,4.,0.]],requires_grad=True);right=(left.detach()+torch.tensor([[0.,1.,0.],[1.,0.,0.]])).requires_grad_();loss=js(torch.log_softmax(left,-1),torch.log_softmax(right,-1)).mean();grads=torch.autograd.grad(loss,[left,right]);assert all(torch.isfinite(g).all() and torch.count_nonzero(g)>0 for g in grads)
    same=torch.log_softmax(left,-1);assert torch.allclose(js(same,same),torch.zeros(2),atol=1e-7) and js(same,same.flip(0)).min()>.1
    result={'identity_js_and_gradient_zero':True,'semantic_action_permutation_aligns':True,'other_probability_mass_retained':True,'both_gradient_paths_nonzero_when_disagreeing':True,'different_history_sides_not_conflated':True,'no_model_optimizer_updates':True};(R/'CONSISTENCY_MATH_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
