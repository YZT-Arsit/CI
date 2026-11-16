"""Learned local-rule distillation control and separated safety/value heads.

The max-min composition operator is an explicit architectural inductive bias.
This is NOT a free-text LLM or a newly claimed method. Relation gates and endpoint
match parameters are learned; no exact closure or task oracle is called here.
"""
import hashlib,random
import numpy as np
import torch
from torch import nn
from .receipt_policy import candidate_view,GraphPolicy


def facts_and_flags(obs,action):
    _,edges,flags=candidate_view(obs,action)
    # The old encoder omitted authorized future obligations at Source states.
    # A candidate COMMIT/DELIVER obligates only its selected provider's fields.
    if action.startswith('COMMIT:') or action=='DELIVER_INPUTS':
        ticket=action.split(':',1)[1] if action.startswith('COMMIT:') else obs['committed_ticket']
        edges['required_future_link']=next(p['links'] for p in obs['contract']['providers'] if p['ticket']==ticket) if obs['contract']['binding_required'] else []
    facts=[(kind,a,b) for kind,name in enumerate(('identity','actual_link','required_future_link','history')) for a,b in edges[name]]
    return facts,np.asarray(flags,np.float32)

def encode(obs,action,*,salt=0,max_facts=128,width=16):
    facts,flags=facts_and_flags(obs,action)
    if len(facts)>max_facts:raise ValueError('fact capacity exceeded; no truncation')
    def vector(name):
        seed=int(hashlib.sha256((name+':'+str(salt)).encode()).hexdigest()[:16],16)
        v=np.random.default_rng(seed).standard_normal(width);return (v/np.sqrt(np.mean(v*v))).astype(np.float32)
    values={v:vector(v) for _,a,b in facts for v in (a,b)}
    types=np.zeros(max_facts,np.int64);head=np.zeros((max_facts,width),np.float32);tail=head.copy();valid=np.zeros(max_facts,np.float32)
    for i,(t,a,b) in enumerate(facts):types[i]=t;head[i]=values[a];tail[i]=values[b];valid[i]=1
    return types,head,tail,valid,flags

def collate(items):
    maximum=max(1,max(int(x[3].sum()) for x in items))
    return [torch.tensor(np.array([x[i][:maximum] if i<4 else x[i] for x in items])) for i in range(5)]

class LocalRuleCore(nn.Module):
    def __init__(self,no_binding=False):
        super().__init__();self.gates=nn.Parameter(torch.randn(2,3,4)*.1)
        self.match_bias=nn.Parameter(torch.zeros(()));self.match_scale=nn.Parameter(torch.zeros(()));self.no_binding=no_binding

    def forward(self,types,head,tail,valid):
        distance=(tail[:,:,None,:]-head[:,None,:,:]).square().mean(-1)
        match=torch.sigmoid(self.match_bias-torch.nn.functional.softplus(self.match_scale)*distance)
        if self.no_binding:match=torch.ones_like(match)
        match=match*valid[:,:,None]*valid[:,None,:];gate=torch.sigmoid(self.gates);risks=[]
        for query in range(2):
            start,cont,end=[gate[query,k][types]*valid for k in range(3)]
            reach=start
            transition=torch.minimum(match,cont[:,None,:])
            for _ in range(types.shape[1]):
                reach=torch.maximum(reach,torch.minimum(reach[:,:,None],transition).amax(1))
            terminal=torch.minimum(torch.minimum(reach[:,:,None],match).amax(1),end)
            risks.append(terminal.amax(1))
        return torch.stack(risks,dim=1)

    def distillation_loss(self,rng,batch=128):
        # Training-only primitive norm/proof supervision. Not an inference mask.
        target=torch.tensor([[[1,0,0,0],[0,1,0,0],[0,0,0,1]],
                             [[1,0,0,0],[0,1,1,0],[0,0,0,1]]],dtype=torch.float32)
        gate_loss=torch.nn.functional.binary_cross_entropy_with_logits(self.gates,target)
        left=torch.tensor(rng.standard_normal((batch,16)),dtype=torch.float32);left=left/left.square().mean(1,keepdim=True).sqrt()
        right=torch.tensor(rng.standard_normal((batch,16)),dtype=torch.float32);right=right/right.square().mean(1,keepdim=True).sqrt()
        same=torch.tensor(rng.integers(0,2,batch),dtype=torch.float32);right=torch.where(same[:,None].bool(),left,right)
        distance=(left-right).square().mean(-1)
        return gate_loss+torch.nn.functional.binary_cross_entropy_with_logits(self.match_bias-torch.nn.functional.softplus(self.match_scale)*distance,same)

class LocalRulePolicy(nn.Module):
    def __init__(self,no_binding=False):
        super().__init__();self.core=LocalRuleCore(no_binding)
        self.head=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,types,head,tail,valid,flags):
        risk=self.core(types,head,tail,valid)
        return self.head(torch.cat([flags,risk],dim=1))
    @staticmethod
    def score(logits):
        probabilities=torch.sigmoid(logits)
        return 4*probabilities[:,0]+probabilities[:,1]

class DenseGraphPolicy(GraphPolicy):
    def __init__(self):
        super().__init__();self.head[-1]=nn.Linear(16,2)
    score=staticmethod(LocalRulePolicy.score)

def graph_encode(obs,action,max_nodes=128):
    facts,flags=facts_and_flags(obs,action);subject=obs['subject']
    nodes=[subject]+sorted({v for _,a,b in facts for v in (a,b)}-{subject})
    if len(nodes)>max_nodes:raise ValueError('node capacity exceeded')
    index={v:i for i,v in enumerate(nodes)};x=np.zeros((max_nodes,12),np.float32);x[:len(nodes),2:]=flags;x[0,0]=1
    adjacency=np.zeros((4,max_nodes,max_nodes),np.float32)
    for kind,a,b in facts:
        adjacency[kind,index[a],index[b]]=1
        if kind==3:x[index[b],1]=1
    return x,adjacency
