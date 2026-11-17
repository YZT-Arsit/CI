"""Supervised two-view loss with semantic, conditional consistency."""
import math,random,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'20261117_action_binding'))
from coverage_repair import torch,answer_ids,forward_targets,response
sys.path.insert(0,str(HERE))
from render import query,actor

def semantic_log_probs(logits,token_ids):
    """Retain all probability mass: named semantic tokens plus one OTHER bin."""
    indices=torch.as_tensor(token_ids,dtype=torch.long,device=logits.device)
    assert indices.ndim==1 and len(indices)==len(set(indices.tolist()))
    mask=torch.zeros(logits.shape[-1],dtype=torch.bool,device=logits.device);mask[indices]=True
    bins=torch.cat([logits[...,indices],torch.logsumexp(logits[...,~mask],dim=-1,keepdim=True)],dim=-1)
    return torch.log_softmax(bins,dim=-1)

def js(left,right):
    mixture=torch.logaddexp(left,right)-math.log(2.)
    return .5*((left.exp()*(left-mixture)).sum(-1)+(right.exp()*(right-mixture)).sum(-1))

def loss_for_views(tokenizer,model,pair,pool,views,step,weight):
    assert len(pair)==len(views)==2
    letters=answer_ids(tokenizer);binary_ids=[tokenizer.encode(str(i),add_special_tokens=False)[0] for i in (0,1)];binary=dict(zip(binary_ids,['0','1']));texts=[];answers=[];indices=[];labels=[]
    for view,seeds in enumerate(views):
        for side,row in enumerate(pair):
            for candidate in row['candidates']:
                texts.append(query(row['observation'],pool,seeds,candidate['action']));answers.append('ANSWER:\n'+str(candidate['safe_completion']));indices.append((view,side,candidate['action']));labels.append(candidate['safe_completion'])
    qlogits,qce,qtokens=forward_targets(tokenizer,model,texts,answers);estimates=[[{},{}],[{},{}]]
    for (view,side,action),value in zip(indices,qlogits[:,-2].detach().argmax(-1).tolist()):estimates[view][side][action]=binary.get(value,'unknown')
    positive=torch.tensor(labels,dtype=torch.bool);bit=.5*(qce[positive,-2].mean()+qce[~positive,-2].mean()) if positive.any() and (~positive).any() else qce[:,-2].mean();qloss=bit+.1*torch.cat([qce[:,:-2],qce[:,-1:]],1).mean();half=len(indices)//2;assert [(s,a) for v,s,a in indices[:half]]==[(s,a) for v,s,a in indices[half:]];qprob=semantic_log_probs(qlogits[:,-2],binary_ids);qjs=js(qprob[:half],qprob[half:]).mean();texts=[];answers=[];goodsets=[];orders=[]
    for view,seeds in enumerate(views):
        for side,row in enumerate(pair):
            text,order=actor(row['observation'],pool,seeds,estimates[view][side]);answer,good=response(row,order,pool,seeds['names'],'plain',random.Random(step*2+side));texts.append(text);answers.append(answer);goodsets.append(good);orders.append(order)
    logits,ce,atokens=forward_targets(tokenizer,model,texts,answers);aloss=torch.stack([torch.logsumexp(logits[i,-2],0)-torch.logsumexp(logits[i,-2,letters[good]],0) for i,good in enumerate(goodsets)]).mean()+.1*torch.cat([ce[:,:-2],ce[:,-1:]],1).mean();aprobs=[]
    for i,order in enumerate(orders):
        canonical=pair[i%2]['observation']['actions'];aprobs.append(semantic_log_probs(logits[i,-2],[letters[order.index(action)].item() for action in canonical]))
    eligible=[side for side in (0,1) if estimates[0][side]==estimates[1][side]]
    ajs=torch.stack([js(aprobs[side],aprobs[2+side]) for side in eligible]).mean() if eligible else logits.sum()*0.
    total=qloss+aloss+weight*(qjs+ajs)
    return total,{'query_loss':qloss,'actor_loss':aloss,'query_js':qjs,'actor_js':ajs},qtokens+atokens,estimates,len(eligible)
