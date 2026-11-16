"""Bounded NumPy DPO optimizer mathematics on the existing 38x280 design.

No LLM, autograd, distributed backend, mixed-precision model kernel or new data.
Each diagnostic executes in a separate worker with a hard 9.2 s wall limit.
"""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from experiments.preference_probe import features

SEED=1729
BETA=.7
PROBES=('microbatch','ddp_counts','clipping','adam_resume','reference','precision','shared_roles')


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def digest(array):return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_inputs(dataset):
    rows=[json.loads(line) for line in Path(dataset).read_text(encoding='utf-8').splitlines()]
    x=np.stack([features(r['prompt'],r['chosen'])-features(r['prompt'],r['rejected']) for r in rows])
    if x.shape!=(38,280):raise ValueError('Frozen experiment requires the existing 38x280 design')
    rng=np.random.default_rng(SEED)
    reference=rng.normal(0,.02,x.shape[1]);theta=reference+rng.normal(0,.1,x.shape[1])
    return rows,x,reference,theta


def loss_gradient(x,theta,reference,beta=BETA):
    z=beta*(x@(theta-reference))
    losses=np.logaddexp(0,-z)
    gradients=(-beta*np.exp(-np.logaddexp(0,z)))[:,None]*x
    return float(np.mean(losses)),np.mean(gradients,axis=0),gradients


def comparison(a,b):
    return {'max_abs':float(np.max(np.abs(a-b))),'l2':float(np.linalg.norm(a-b))}


def run_probe(name,dataset,output):
    started=time.perf_counter();rows,x,ref,theta=load_inputs(dataset);n=len(x)
    _,full,per_example=loss_gradient(x,theta,ref)
    groups=[np.arange(i,min(i+8,n)) for i in range(0,n,8)]
    result={}
    if name=='microbatch':
        means=[loss_gradient(x[g],theta,ref)[1] for g in groups]
        weighted=sum(len(g)*m for g,m in zip(groups,means))/n
        wrong=np.mean(means,axis=0)
        result={'microbatch_sizes':[len(g) for g in groups],
                'weighted_vs_full':comparison(weighted,full),'mean_of_batch_means_vs_full':comparison(wrong,full),
                'pass':bool(np.max(np.abs(weighted-full))<1e-12 and np.linalg.norm(wrong-full)>1e-8)}
    elif name=='ddp_counts':
        shards=(np.arange(17),np.arange(17,30),np.arange(30,38))
        means=[loss_gradient(x[g],theta,ref)[1] for g in shards]
        # Simulated DDP averages rank gradients: each local SUM is scaled by world_size/global_count.
        normalized=np.mean([len(shards)*len(g)*m/n for g,m in zip(shards,means)],axis=0)
        wrong=np.mean(means,axis=0)
        result={'rank_sample_counts':[len(g) for g in shards],'global_count':n,
                'global_count_normalized_vs_full':comparison(normalized,full),
                'mean_of_rank_means_vs_full':comparison(wrong,full),
                'pass':bool(np.max(np.abs(normalized-full))<1e-12 and np.linalg.norm(wrong-full)>1e-8),
                'scope':'algebraic simulation; no DDP process group or accelerator'}
    elif name=='clipping':
        limit=.05
        def clip(g):return g*min(1.,limit/max(float(np.linalg.norm(g)),1e-300))
        accumulated=clip(full)
        premature=sum(len(g)*clip(loss_gradient(x[g],theta,ref)[1]) for g in groups)/n
        result={'max_norm':limit,'unclipped_global_norm':float(np.linalg.norm(full)),
                'clip_after_accumulation_norm':float(np.linalg.norm(accumulated)),
                'clip_each_microbatch_then_accumulate_norm':float(np.linalg.norm(premature)),
                'difference':comparison(accumulated,premature),
                'pass':bool(np.linalg.norm(accumulated-premature)>1e-8)}
    elif name=='adam_resume':
        total=48;split=19;batch_size=7;lr=.01;b1=.9;b2=.999;eps=1e-8
        initial=ref.copy()
        def train(parameters,m,v,t,rng,steps):
            batches=[]
            for _ in range(steps):
                indices=rng.choice(n,batch_size,replace=True);batches.append(indices.tolist())
                _,gradient,_=loss_gradient(x[indices],parameters,ref)
                t+=1;m=b1*m+(1-b1)*gradient;v=b2*v+(1-b2)*gradient*gradient
                parameters=parameters-lr*(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+eps)
            return parameters,m,v,t,batches
        continuous_rng=np.random.default_rng(SEED+41)
        continuous=train(initial.copy(),np.zeros_like(ref),np.zeros_like(ref),0,continuous_rng,total)
        interrupted_rng=np.random.default_rng(SEED+41)
        checkpoint=train(initial.copy(),np.zeros_like(ref),np.zeros_like(ref),0,interrupted_rng,split)
        root=Path(output).parent
        np.savez(root/'adam_checkpoint.npz',theta=checkpoint[0],m=checkpoint[1],v=checkpoint[2],t=checkpoint[3],frozen_reference=ref)
        save(root/'adam_rng_state.json',deepcopy(interrupted_rng.bit_generator.state))
        with np.load(root/'adam_checkpoint.npz') as stored:
            restored=[stored[k].copy() for k in ('theta','m','v')]+[int(stored['t'])]
        rng_state=json.loads((root/'adam_rng_state.json').read_text(encoding='utf-8'))
        def restored_rng():
            rng=np.random.default_rng();rng.bit_generator.state=deepcopy(rng_state);return rng
        resumed_rng=restored_rng();resumed=train(*restored,resumed_rng,total-split)
        reset_optimizer=train(restored[0].copy(),np.zeros_like(ref),np.zeros_like(ref),0,restored_rng(),total-split)
        reset_rng=train(*[v.copy() if isinstance(v,np.ndarray) else v for v in restored],
                        np.random.default_rng(SEED+41),total-split)
        np.savez(root/'adam_final_states.npz',continuous=continuous[0],resumed=resumed[0],
                 reset_optimizer=reset_optimizer[0],reset_rng=reset_rng[0])
        exact=all(np.array_equal(a,b) for a,b in zip(continuous[:4],resumed[:4]))
        same_batches=continuous[4]==checkpoint[4]+resumed[4]
        same_rng=continuous_rng.bit_generator.state==resumed_rng.bit_generator.state
        result={'total_steps':total,'checkpoint_step':split,'batch_size':batch_size,
                'optimizer':{'name':'explicit_float64_Adam','lr':lr,'beta1':b1,'beta2':b2,'epsilon':eps},
                'theta_m_v_t_bitwise_equal':exact,'rng_state_equal':same_rng,'batch_schedule_equal':same_batches,
                'continuous_theta_sha256':digest(continuous[0]),'resumed_theta_sha256':digest(resumed[0]),
                'reset_optimizer_difference':comparison(continuous[0],reset_optimizer[0]),
                'reset_rng_difference':comparison(continuous[0],reset_rng[0]),
                'continuous_batch_indices':continuous[4],
                'pass':bool(exact and same_batches and same_rng and np.linalg.norm(continuous[0]-reset_optimizer[0])>1e-8
                            and np.linalg.norm(continuous[0]-reset_rng[0])>1e-8)}
    elif name=='reference':
        steps=40;lr=.2;correct=ref.copy();refreshed=ref.copy();traces=[]
        for step in range(steps):
            fixed_loss,fixed_grad,_=loss_gradient(x,correct,ref)
            # This is a deliberately different algorithm: detached reference reset to theta each step.
            moving_reference=refreshed.copy()
            moving_loss,moving_grad,_=loss_gradient(x,refreshed,moving_reference)
            correct-=lr*fixed_grad;refreshed-=lr*moving_grad
            traces.append({'step':step,'fixed_reference_objective':fixed_loss,'refreshed_reference_objective':moving_loss})
        result={'steps':steps,'learning_rate':lr,'loss_trace':traces,'parameter_difference':comparison(correct,refreshed),
                'fixed_run_final_loss_under_original_reference':loss_gradient(x,correct,ref)[0],
                'refreshed_run_final_loss_under_original_reference':loss_gradient(x,refreshed,ref)[0],
                'wrong_variant_is_not_standard_fixed_reference_DPO':True,
                'pass':bool(np.linalg.norm(correct-refreshed)>1e-8 and traces[-1]['fixed_reference_objective']<traces[0]['fixed_reference_objective']
                            and all(abs(t['refreshed_reference_objective']-np.log(2))<1e-12 for t in traces))}
    elif name=='precision':
        baseline_loss,baseline_gradient,_=loss_gradient(x,theta,ref)
        actual=[];stress=[];stress_margins=np.array([-10000,-1000,-100,-20,-2,0,2,20,100,1000,10000])
        for dtype in (np.float64,np.float32,np.float16):
            xd=x.astype(dtype);td=theta.astype(dtype);rd=ref.astype(dtype);beta=np.array(BETA,dtype=dtype)
            with np.errstate(over='ignore',under='ignore',invalid='ignore',divide='ignore'):
                z=beta*(xd@(td-rd));losses=np.logaddexp(np.array(0,dtype=dtype),-z)
                grad=np.mean((-beta*np.exp(-np.logaddexp(np.array(0,dtype=dtype),z)))[:,None]*xd,axis=0)
                raw=stress_margins.astype(dtype)
                naive=np.log(np.array(1,dtype=dtype)+np.exp(-raw))
                stable=np.logaddexp(np.array(0,dtype=dtype),-raw)
            actual.append({'dtype':np.dtype(dtype).name,'loss':float(np.mean(losses)),
                           'loss_abs_error_vs_float64':abs(float(np.mean(losses))-baseline_loss),
                           'gradient_error_vs_float64':comparison(grad.astype(float),baseline_gradient),
                           'finite_gradient':bool(np.isfinite(grad).all())})
            stress.append({'dtype':np.dtype(dtype).name,'naive_finite_count':int(np.isfinite(naive).sum()),
                           'stable_finite_count':int(np.isfinite(stable).sum()),
                           'naive_loss':[float(v) if np.isfinite(v) else None for v in naive],
                           'stable_loss':[float(v) for v in stable]})
        result={'actual_design':actual,'synthetic_margins':stress_margins.tolist(),'synthetic_stress':stress,
                'scope':'NumPy dtype operations only; NOT BF16, autocast, GPU kernels, or real LLM numeric validation',
                'pass':bool(all(r['finite_gradient'] for r in actual) and all(r['stable_finite_count']==len(stress_margins)
                             and r['naive_finite_count']<len(stress_margins) for r in stress))}
    elif name=='shared_roles':
        roles=np.array([json.loads(row['prompt'])['role'] for row in rows])
        counts={role:int(np.sum(roles==role)) for role in sorted(set(roles))}
        role_sums={role:per_example[roles==role].sum(axis=0) for role in counts}
        combined=sum(role_sums.values())/n
        wrong=np.mean([role_sums[r]/counts[r] for r in counts],axis=0)
        result={'role_sample_counts':counts,'sum_role_gradients_over_global_count_vs_full':comparison(combined,full),
                'mean_of_role_means_vs_full':comparison(wrong,full),
                'shared_parameter_dimensions':x.shape[1],
                'pass':bool(np.max(np.abs(combined-full))<1e-12),
                'scope':'linearity for one shared finite theta and sample-weighted objective, not cross-role LLM efficacy'}
    result.update(probe=name,seed=SEED,beta=BETA,rows=n,dimensions=x.shape[1],design_sha256=digest(x),
                  dataset_sha256=sha256(Path(dataset).read_bytes()).hexdigest(),seconds=time.perf_counter()-started,
                  evidence_class='finite_numpy_DPO_implementation_diagnostic_NOT_real_model_training')
    save(output,result);print(json.dumps({'probe':name,'pass':result['pass'],'seconds':result['seconds']}))
    return result['pass']


def orchestrate(dataset,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('Use a new destination; no diagnostic rerun selection')
    rows,x,ref,theta=load_inputs(dataset)
    files=[Path(__file__),Path(__file__).with_name('preference_probe.py'),Path(dataset)]
    save(root/'manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'beta':BETA,
        'probes':list(PROBES),'rows':len(rows),'dimensions':x.shape[1],'timeout_seconds':9.2,
        'source_and_input_sha256':{str(p):sha256(p.read_bytes()).hexdigest() for p in files},
        'design_sha256':digest(x),'scope':'mathematical diagnostics on frozen existing preferences; no model or data expansion',
        'numpy_version':np.__version__,'python_version':sys.version})
    np.save(root/'design.npy',x);(root/'source_snapshot.py').write_bytes(Path(__file__).read_bytes())
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    receipts=[]
    for probe in PROBES:
        command=[sys.executable,'-B','-m','experiments.optimizer_probe','--dataset',str(dataset),'--probe',probe,
                 '--output',str(root/(probe+'.json'))]
        t=time.perf_counter()
        try:
            proc=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=9.2,env=environment)
            receipt={'probe':probe,'status':'completed' if proc.returncode==0 else 'failed','returncode':proc.returncode,
                     'stdout':proc.stdout,'stderr':proc.stderr}
        except subprocess.TimeoutExpired as exc:receipt={'probe':probe,'status':'timeout','error':str(exc)}
        receipt['worker_seconds']=time.perf_counter()-t;receipts.append(receipt)
        print(json.dumps(receipt),flush=True)
    save(root/'worker_receipts.json',receipts)
    result={'workers':len(receipts),'completed':sum(r['status']=='completed' for r in receipts),
            'timeout_count':sum(r['status']=='timeout' for r in receipts),
            'max_worker_seconds':max(r['worker_seconds'] for r in receipts),
            'checks':{p:json.loads((root/(p+'.json')).read_text(encoding='utf-8'))['pass'] for p in PROBES if (root/(p+'.json')).exists()}}
    save(root/'summary.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--probe',choices=PROBES);args=parser.parse_args()
    if args.probe:raise SystemExit(0 if run_probe(args.probe,args.dataset,args.output) else 1)
    orchestrate(args.dataset,args.output)
