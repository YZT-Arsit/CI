"""Fresh confirmation of direct action-label learning, including matched interventions."""
from lm_common import *
import time,hashlib
from kci.relation_solution import LocalRulePolicy
from experiments.relation_solution_data import heldout_world,intervene
from experiments.receipt_dataset import make_world
from representation_repair import execute
from train_structured import short
from kci.dynamic_tasks import canonical

def run():
    out=R/'direct_learning_evaluation';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    splits=['test','depth4','depth6','depth10','noise','optional','authorized','draft','both','broken','reverse','branch','cycle']
    plan={'models':'all three direct-learning checkpoints; no selection','splits':splits,'worlds_per_split':64,'new_seed_offset':60000000,'width':64,'max_facts':256,'extra_controls':['paired test/depth4/depth6/depth10 swaps','fresh vector salt 990','external complete'],
        'scope':'fresh confirmatory structured experiment; fixed max-min architecture remains supplied; not a language-model result','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plan.json').write_text(json.dumps(plan,indent=2));worldsets={}
    for split in splits:
        worlds=[]
        for i in range(64):
            base=heldout_world(split,i);seed=int(base['world_id'].split('_')[-1])+60000000
            w=make_world(seed,base['kind'],depth={'depth4':4,'depth6':6,'depth10':10}.get(split,2),noise=12 if split=='noise' else 2,
                allow=split=='authorized',required=split!='optional',delivered=split!='draft',mode='both' if split=='both' else 'one');w['base_group_id']=w['world_id']
            if split in ('broken','reverse','branch','cycle'):w=intervene(w,split)
            worlds.append(w)
        worldsets[split]=worlds
    for base in ('test','depth4','depth6','depth10'):worldsets[base+'_swap']=[intervene(w,'swap') for w in worldsets[base]]
    for split in ('external_complete','external_complete_swap'):
        worldsets[split]=[json.loads(l) for p in sorted((R/'dataset'/split).glob('worlds_*.jsonl')) for l in p.read_text().splitlines()]
    for split,ws in worldsets.items():(out/(split+'_worlds.jsonl')).write_text(''.join(canonical(w)+'\n' for w in ws))
    seal=R/'auxiliary_ablation/RESULTS.json'
    while not seal.exists():time.sleep(2)
    summaries=[]
    for report in json.loads(seal.read_text())['models']:
        seed=report['seed'];path=R/'auxiliary_ablation'/f'model_{seed}.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256']
        model=LocalRulePolicy();model.load_state_dict(torch.load(path,weights_only=True));model.eval();folder=out/f'direct_{seed}';folder.mkdir()
        for split,ws in worldsets.items():
            ledger=[];rows=[short(lambda:execute(model,w,64),ledger,'episode') for w in ws]
            (folder/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
            summary={'model':folder.name,'condition':split,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('ci','safe_success','refused')},'max_episode_seconds':max(r['seconds'] for r in ledger)};summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'summaries':summaries,'training_updates_matched_to_primary':False,'primitive_teacher_used':False},indent=2))

if __name__=='__main__':run()
