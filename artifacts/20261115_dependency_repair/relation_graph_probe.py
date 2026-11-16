"""Train a tiny generic typed message-passing classifier; not an LLM policy."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
from hashlib import sha256
import argparse,json
import numpy as np

R=Path(__file__).resolve().parent
D=8


def graphs(seed,n,branching=False):
    rng=np.random.default_rng(seed);xs=[];adj=[];roots=[];labels=[];meta=[]
    for allowed in (False,True):
      for match in (False,True):
       for i in range(64):
        # person, n aliases, n cases, sensitive value; no risk/closure feature.
        size=2*n+2;root=0;sensitive=size-1;alias=int(rng.integers(n))
        permutation=rng.permutation(n)
        neighbors={j:{int(permutation[j])} for j in range(n)}
        if branching:
            for j in range(n):neighbors[j].add(int(permutation[(j+1)%n]))
        candidates=sorted(neighbors[alias] if match else set(range(n))-neighbors[alias])
        history_case=int(rng.choice(candidates))
        a=np.zeros((3,size,size));a[0,root,1+alias]=1
        for j,targets in neighbors.items():
            for c in targets:a[1,1+j,1+n+c]=1
        a[2,1+n+history_case,sensitive]=1
        x=np.zeros((size,D));x[:,0]=1;x[sensitive,1]=1;x[:,2]=int(allowed)
        rootmask=np.zeros(size);rootmask[root]=1
        order=rng.permutation(size);x=x[order];a=a[:,order][:,:,order];rootmask=rootmask[order]
        # Independent truth check by enumerating the declared three joins.
        closure=bool(np.any((a[0]@a[1]@a[2])[np.argmax(rootmask)]))
        assert closure==match
        xs.append(x);adj.append(a);roots.append(rootmask);labels.append(float(match and not allowed))
        meta.append({'allowed':allowed,'match':match,'n':n,'branching':branching})
    return tuple(np.array(x) for x in (xs,adj,roots,labels)),meta


def initialize(depth,seed):
    rng=np.random.default_rng(seed)
    return {'self':rng.normal(0,.2,(depth,D,D)),'edge':rng.normal(0,.2,(depth,3,D,D)),
            'bias':np.zeros((depth,D)),'out':rng.normal(0,.2,D),'offset':np.array(0.)}


def forward(p,data):
    x,a,root,_=data;h=x;cache=[]
    for level in range(len(p['self'])):
        messages=np.einsum('brij,bjd->brid',a,h)
        z=h@p['self'][level]+np.einsum('brni,rij->bnj',messages,p['edge'][level])+p['bias'][level]
        after=np.tanh(z);cache.append((h,messages,after));h=after
    pooled=np.einsum('bn,bnd->bd',root,h)
    return pooled@p['out']+p['offset'],(cache,pooled)


def loss_grad(p,data):
    x,a,root,y=data;logit,(cache,pooled)=forward(p,data)
    weights=np.where(y==1,3.,1.);prob=1/(1+np.exp(-np.clip(logit,-60,60)))
    loss=np.mean(weights*(np.logaddexp(0,logit)-y*logit));d=(prob-y)*weights/len(y)
    g={k:np.zeros_like(v) for k,v in p.items()};g['out']=pooled.T@d;g['offset']=np.array(d.sum())
    dh=root[:,:,None]*d[:,None,None]*p['out'][None,None,:]
    for level in reversed(range(len(cache))):
        previous,messages,after=cache[level];dz=dh*(1-after**2)
        g['self'][level]=np.einsum('bni,bnj->ij',previous,dz)
        g['edge'][level]=np.einsum('brni,bnj->rij',messages,dz)
        g['bias'][level]=dz.sum(axis=(0,1))
        dm=np.einsum('bnj,rij->brni',dz,p['edge'][level])
        dh=dz@p['self'][level].T+np.einsum('brij,brid->bjd',a,dm)
    return loss,g


def audit_gradient(p,data):
    loss,g=loss_grad(p,data);errors=[]
    for name,index in [('self',(0,1,2)),('edge',(1,2,1,3)),('bias',(2,2)),('out',(3,)),('offset',())]:
        original=float(p[name][index]);h=1e-5
        p[name][index]=original+h;plus=loss_grad(p,data)[0]
        p[name][index]=original-h;minus=loss_grad(p,data)[0]
        p[name][index]=original;errors.append(abs((plus-minus)/(2*h)-g[name][index]))
    assert max(errors)<1e-6
    return max(errors)


def run(arm,seed,output):
    depth={'depth0':0,'depth2':2,'depth3':3,'shuffled':3}[arm]
    train,trainmeta=graphs(51001,2);rng=np.random.default_rng(seed)
    if arm=='shuffled':train=(*train[:3],rng.permutation(train[3]))
    p=initialize(depth,seed);m={k:np.zeros_like(v) for k,v in p.items()};v={k:np.zeros_like(x) for k,x in p.items()}
    gradient_error=audit_gradient(initialize(3,991),tuple(x[:8] for x in train))
    losses=[]
    for t in range(1,301):
        indices=rng.integers(0,len(train[3]),32);batch=tuple(x[indices] for x in train)
        loss,g=loss_grad(p,batch)
        for name in p:
            m[name]=.9*m[name]+.1*g[name];v[name]=.999*v[name]+.001*g[name]**2
            p[name]-=.02*(m[name]/(1-.9**t))/(np.sqrt(v[name]/(1-.999**t))+1e-8)
        if t%50==0:losses.append(float(loss))
    evaluations={}
    for name,n,branch,ds in [('fresh_binding',2,False,62001),('larger',5,False,62002),('branching',5,True,62003)]:
        data,meta=graphs(ds,n,branch);logits=forward(p,data)[0];pred=logits>=0;y=data[3].astype(bool)
        order=np.random.default_rng(77).permutation(data[0].shape[1])
        permuted=(data[0][:,order],data[1][:,:,order][:,:,:,order],data[2][:,order],data[3])
        difference=float(np.max(np.abs(logits-forward(p,permuted)[0])))
        assert difference<1e-10
        strata={}
        for allow in (False,True):
          for match in (False,True):
            indices=[i for i,r in enumerate(meta) if r['allowed']==allow and r['match']==match]
            strata[f'allow{int(allow)}_match{int(match)}']={'correct':int(np.sum(pred[indices]==y[indices])),'total':len(indices)}
        evaluations[name]={'correct':int(np.sum(pred==y)),'total':len(y),'strata':strata,
            'logits':logits.tolist(),'labels':y.tolist(),'node_permutation_max_error':difference,
            'input_sha256':sha256(b''.join(x.tobytes() for x in data)).hexdigest()}
    out=Path(output);out.mkdir(parents=True,exist_ok=True);np.savez(out/'weights.npz',**p)
    gate=all(s['correct']/s['total']>=.95 for e in evaluations.values() for s in e['strata'].values())
    result={'arm':arm,'seed':seed,'steps':300,'batch':32,'learning_rate':.02,'width':D,
       'training_rows':256,'training_input_sha256':sha256(b''.join(x.tobytes() for x in train)).hexdigest(),
       'gradient_max_error':gradient_error,'loss_trace':losses,'evaluations':evaluations,
       'diagnostic_gate_passed':gate,'weights_sha256':sha256((out/'weights.npz').read_bytes()).hexdigest(),
       'scope':'synthetic relation classifier, not a DynamicTask action policy or LLM training',
       'name_invariance':'architectural exact-entity graph representation, not learned token matching',
       'labels_or_closure_in_features':False,'true_source_holdout':False}
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'arm':arm,'seed':seed,'accuracy':{k:[e['correct'],e['total']] for k,e in evaluations.items()},'gate':gate}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm');p.add_argument('seed',type=int);p.add_argument('output');a=p.parse_args();run(a.arm,a.seed,a.output)
