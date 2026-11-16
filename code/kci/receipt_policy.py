"""Shared CPU action scorers. Neither model receives closure or a safety mask."""
import hashlib,random
import numpy as np
import torch
from torch import nn

FEATURES=12
CHANNELS=('identity','actual_link','required_future_link','history')


def candidate_view(obs,action):
    edges={k:[] for k in CHANNELS};subject=obs['subject'];contract=obs['contract']
    for m in obs['delivered']:
        if m['recipient']!='R':continue
        body=m['content'];edges['identity']+=body.get('identities',[])
        edges['actual_link']+=body.get('links',[]);edges['history']+=body.get('histories',[])
    if action.startswith('COMMIT:'):
        edges['identity'].append([subject,action.split(':',1)[1]])
        if contract['binding_required']:
            edges['required_future_link']=[edge for p in contract['providers'] for edge in p['links']]
    if action=='PUBLISH_FULL':
        provider=next(p for p in contract['providers'] if p['ticket']==obs['committed_ticket'])
        edges['actual_link']+=provider['links']
    role=['Planner','Source','Worker'].index(obs['role'])
    flags=[int(contract['permission']),int(contract['binding_required'])]+[int(role==i) for i in range(3)]+[
        int(action.startswith('COMMIT:')),int(action=='DELIVER_INPUTS'),int(action.startswith('PUBLISH_')),
        int(action=='PUBLISH_FULL'),int(action=='WITHHOLD')]
    return subject,edges,flags


def graph_encode(obs,action,max_nodes=40):
    subject,edges,flags=candidate_view(obs,action)
    nodes=[subject]+sorted({v for pairs in edges.values() for pair in pairs for v in pair}-{subject})
    if len(nodes)>max_nodes:raise ValueError('graph exceeds declared padded size')
    lookup={v:i for i,v in enumerate(nodes)};x=np.zeros((max_nodes,FEATURES),np.float32)
    x[:len(nodes),2:]=flags;x[0,0]=1
    for _,v in edges['history']:x[lookup[v],1]=1
    adjacency=np.zeros((4,max_nodes,max_nodes),np.float32)
    for channel,name in enumerate(CHANNELS):
        for a,b in edges[name]:adjacency[channel,lookup[a],lookup[b]]=1
    return x,adjacency


def token_encode(obs,action,*,fresh=False,max_tokens=192,shuffle_seed=0):
    # Generic entity tokenization; no computed equality edges or graph closure.
    # Repeated raw entity strings receive the same token, as with ordinary atomic
    # symbol tokens. Test fresh=True uses entirely unused entity embedding rows.
    subject,edges,flags=candidate_view(obs,action)
    nodes=sorted({subject}|{v for pairs in edges.values() for pair in pairs for v in pair})
    seed=int(hashlib.sha256(('|'.join(nodes)+str(shuffle_seed)).encode()).hexdigest()[:12],16)
    rng=random.Random(seed);pool=list(range(128 if fresh else 64,192 if fresh else 128));rng.shuffle(pool)
    mapping=dict(zip(nodes,pool));triples=[]
    for channel,name in enumerate(CHANNELS):
        for a,b in edges[name]:triples.append([20+channel,mapping[a],mapping[b]])
    rng.shuffle(triples)
    tokens=[1,mapping[subject]]+[3+2*i+int(flag) for i,flag in enumerate(flags)]
    # Relation tokens must not overlap flag tokens (the last flag occupies 21/22).
    triples=[[30+(t[0]-20),t[1],t[2]] for t in triples]
    tokens += [v for triple in triples for v in triple]
    if len(tokens)>max_tokens:raise ValueError('tokens exceed budget; no truncation')
    return np.array(tokens+[0]*(max_tokens-len(tokens)),np.int64)


class GraphPolicy(nn.Module):
    def __init__(self,depth=6,width=16,relations=True):
        super().__init__();self.relations=relations;self.depth=depth
        self.input=nn.Linear(FEATURES,width)
        self.own=nn.Linear(width,width)
        self.edges=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(4))
        self.head=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,1))
    def forward(self,x,a):
        h=torch.tanh(self.input(x))
        if not self.relations:a=torch.zeros_like(a)
        for _ in range(self.depth):
            z=self.own(h)
            for i,transform in enumerate(self.edges):z=z+transform(torch.bmm(a[:,i],h))
            h=torch.tanh(z)
        return self.head(h[:,0]).squeeze(-1)


class TokenPolicy(nn.Module):
    def __init__(self,width=32,layers=4):
        super().__init__();self.embedding=nn.Embedding(192,width,padding_idx=0)
        self.position=nn.Embedding(192,width)
        layer=nn.TransformerEncoderLayer(width,4,dim_feedforward=64,dropout=0.,batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,layers,enable_nested_tensor=False)
        self.head=nn.Linear(width,1)
        # Fixed random entity codes remove per-symbol fitting as an explanation;
        # all attention/FFN/readout and non-entity token embeddings are learned.
        self.embedding.weight.register_hook(self.freeze_entities)
    @staticmethod
    def freeze_entities(gradient):
        gradient=gradient.clone();gradient[64:]=0;return gradient
    def forward(self,ids):
        mask=ids==0;length=int((~mask).sum(1).max());ids=ids[:,:length];mask=mask[:,:length]
        x=self.embedding(ids)+self.position(torch.arange(length,device=ids.device))[None]
        out=self.encoder(x,src_key_padding_mask=mask)
        return self.head(out[:,0]).squeeze(-1)
