"""Matched two-view supervised control versus conditional semantic consistency."""
from probe import *
import copy
from coverage_repair import full_buckets
from two_view_loss import loss_for_views
from build_structural_cases import pair as internal_pair
from build_degree_controls_v2 import anchored_pair

def fresh_dev():
    result={};paired=[]
    for family,count in [('base',8),('depth3',4),('internal3',4),('cycle4',4)]:
        for view in (0,1):
            key=family+'_v'+str(view);result[key]=[];result[key+'_swap']=[];paired.append(key)
        for n in range(count):
            seed=260000000+{'base':0,'depth3':1000,'internal3':2000,'cycle4':3000}[family]+n;kind=('meeting','query')[n%2]
            if family=='internal3':worlds=internal_pair(seed,3,kind)
            elif family=='cycle4':worlds=anchored_pair(seed,4,kind)
            else:
                a=make_world(seed,kind,depth=3 if family=='depth3' else 1,noise=2,mode='one',allow=False,required=True,delivered=True);a['base_group_id']=a['world_id'];worlds=[a,intervene(a,'swap')]
            for view in (0,1):
                for side,world in enumerate(worlds):
                    w=copy.deepcopy(world);w['base_group_id']+='_view'+str(view);result[family+'_v'+str(view)+('_swap' if side else '')].append(w)
    for condition in ['optional','authorized','draft','both']:
        for view in (0,1):result[condition+'_v'+str(view)]=[]
        for n in range(4):
            seed=260010000+{'optional':0,'authorized':1000,'draft':2000,'both':3000}[condition]+n;w=make_world(seed,('meeting','query')[n%2],depth=1,noise=2,mode='both' if condition=='both' else 'one',allow=condition=='authorized',required=condition!='optional',delivered=condition!='draft')
            for view in (0,1):
                v=copy.deepcopy(w);v['base_group_id']=v['world_id']+'_view'+str(view);result[condition+'_v'+str(view)].append(v)
    return result,paired

def run_repair():
    out=HERE/'repair';out.mkdir(exist_ok=True);assert not (out/'PLAN.json').exists();assert json.loads((HERE/'CONSISTENCY_MATH_CHECK.json').read_text())['both_gradient_paths_nonzero_when_disagreeing'];assert json.loads((HERE/'REPAIR_MODEL_SMOKE.json').read_text())['all_roles_passed'];assert json.loads((HERE/'VARIED_MODEL_SMOKE.json').read_text())['all_roles_passed'];buckets=full_buckets();rng=random.Random(26117);batches=[]
    for step in range(80):
        role=('Planner','Source','Worker','Worker')[step%4];keys=sorted(k for k in buckets if k[0]==role);key=rng.choice(keys);batches.append({'key':list(key),'index':rng.randrange(len(buckets[key])),'views':[{k:rng.randrange(10**9) for k in ['names','facts','plans','options','links']} for _ in range(2)]})
    worlds,paired=fresh_dev();data=out/'dev_worlds.json';data.write_text(json.dumps(worlds),encoding='utf-8');assert hashlib.sha256(data.read_bytes()).hexdigest()==json.loads((HERE/'REPAIR_DATA_CHECK.json').read_text())['expected_dev_sha256'];folder=HERE.parent/'20261117_action_binding/coverage_repair/class_balanced';seal=json.loads((folder/'CHECKPOINT_SEALED.json').read_text());plan={'arms':['two_view_sft','conditional_js'],'evidence_arms':['two_view_sft','conditional_js'],'steps':80,'seed':17,'lr':.0008,'batches':batches,'paired_conditions':paired,'initial_adapter_hashes':seal['adapter_hashes'],'consistency_weights':{'two_view_sft':0.,'conditional_js':1.},'query_classes':['0','1','OTHER'],'actor_classes':'semantic action IDs plus OTHER, compared only when model-estimate maps agree','format_weight':.1,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'dependency_hashes':{p:hashlib.sha256((HERE/p).read_bytes()).hexdigest() for p in ['render.py','two_view_loss.py']},'dev_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),'scope':'matched seed17 freshDEV pilot; same worlds reused across two presentations; no source-level holdout or full-goal completion','oracle_at_runtime':False};(out/'PLAN.json').write_text(json.dumps(plan,indent=2));summaries=[]
    for arm in plan['arms']:
        tokenizer,model=setup(17);assert all(hashlib.sha256((folder/'adapter'/p).read_bytes()).hexdigest()==h for p,h in seal['adapter_hashes'].items());set_peft_model_state_dict(model,load_file(str(folder/'adapter/adapter_model.safetensors')));pool=palette(tokenizer);model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0008,weight_decay=0.);target=out/arm;target.mkdir();tokens_total=0
        for step,batch in enumerate(batches):
            def timeout():
                (target/'TIMEOUT.json').write_text(json.dumps({'step':step,'seconds':9.5}));os._exit(124)
            timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();start=time.perf_counter();pair=buckets[tuple(batch['key'])][batch['index']];loss,parts,tokens,estimates,eligible=loss_for_views(tokenizer,model,pair,pool['train'],batch['views'],step,plan['consistency_weights'][arm]);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();timer.cancel();tokens_total+=tokens;record={'step':step+1,**{k:float(v.detach()) for k,v in parts.items()},'input_tokens':tokens,'actor_consistency_eligible_states':eligible,'predicted_estimates':estimates,'seconds':time.perf_counter()-start}
            with (target/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if (step+1)%20==0:print(json.dumps({'arm':arm,**{k:v for k,v in record.items() if k!='predicted_estimates'}}),flush=True)
        model.save_pretrained(target/'adapter');(target/'CHECKPOINT_SEALED.json').write_text(json.dumps({'adapter_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target/'adapter').iterdir() if p.is_file()},'steps':80,'input_tokens':tokens_total},indent=2));model.eval()
        for condition,ws in worlds.items():
            rows=evaluate(tokenizer,model,ws,pool['fresh'],True);path=target/(condition+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');summary={'arm':arm,'condition':condition,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ['safe_success','strict_success','invalid','invalid_queries']},'valid_ci':sum(r['ci'] and not r['invalid'] for r in rows),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};summaries.append(summary);print(json.dumps(summary),flush=True)
        del model,opt;gc.collect()
    (out/'RESULTS.json').write_text(json.dumps({'summaries':summaries,'whole_goal_complete':False},indent=2))

if __name__=='__main__':run_repair()
