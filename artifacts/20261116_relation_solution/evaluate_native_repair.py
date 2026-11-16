"""Transfer unchanged policies to the previous native-output control."""
from evaluate_language import *
import evaluate_language,copy
from kci.native_result_tasks import NativeResultTask
from kci.relation_solution import LocalRulePolicy
from kci.relation_runtime import choose_action
from experiments.receipt_dataset import SOURCE
from train_structured import short

templates={}
def native_env(w):
    source=ROOT/w['source_root_relative'] if 'source_root_relative' in w else SOURCE;key=(w['kind'],str(source))
    if key not in templates:templates[key]=NativeResultTask(w,source)
    e=copy.deepcopy(templates[key]);e.world=copy.deepcopy(w);e.binding['world']=copy.deepcopy(w);return e

def run(kind):
    out=R/(kind+'_native_transfer');out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    n=64 if kind=='structured' else 16
    (out/'plan.json').write_text(json.dumps({'source_worlds':'unchanged previous native_base/native_swapped','first_worlds_per_condition':n,'no_new_binding_field':True,'synthetic_initial_knowledge_and_norm':True,'no_retraining':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    if kind=='language':
        for name in ('language_forced_states','balanced_language_forced_states'):
            while not (R/name/'EVALUATION_SEALED.json').exists():time.sleep(2)
        reports=[(name,r) for name in ('language_models','balanced_language_models') for r in json.loads((R/name/'CHECKPOINTS_SEALED.json').read_text())['models']]
    else:reports=[('structured_models',r) for r in json.loads((R/'structured_models/CHECKPOINTS_SEALED.json').read_text())['models'] if r['arm']=='local']
    evaluate_language.environment=native_env;summaries=[]
    for modeldir,report in reports:
        folder=out/report['model'];folder.mkdir()
        if kind=='language':
            tokenizer,model=setup(report['seed']);adapter=R/modeldir/report['model']/'adapter';assert all(hashlib.sha256((adapter/p).read_bytes()).hexdigest()==h for p,h in report['adapter_hashes'].items())
            set_peft_model_state_dict(model,load_file(str(adapter/'adapter_model.safetensors')));model.eval();pool=palette(tokenizer)['fresh']
        else:
            model=LocalRulePolicy();path=R/modeldir/report['model']/'model.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256'];model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        for split in ('native_base','native_swapped'):
            ws=[json.loads(l) for p in sorted((ROOT/'artifacts/20261115_semantic_repair/dataset'/split).glob('worlds_*.jsonl')) for l in p.read_text().splitlines()][:n]
            for w in ws:w['base_group_id']=w['world_id']
            if kind=='language':rows=evaluate(tokenizer,model,report['arm'],ws,pool)
            else:
                rows=[];ledger=[]
                for w in ws:
                    def episode():
                        e=native_env(w);s=e.initial();trace=[]
                        while not s.done:
                            action,scores=choose_action(model,e.observation(s),seed_for(w['base_group_id'],len(trace)));s,audit=e.transition(s,action);trace.append({'selected':action,'audit':audit,**scores})
                        return {'world_id':w['world_id'],'ci':not s.violation,'safe_success':s.success and not s.violation,'refused':trace[-1]['selected']=='WITHHOLD','trace':trace,'final':asdict(s)}
                    rows.append(short(episode,ledger,'episode'))
            (folder/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows));summary={'model':report['model'],'condition':split,'episodes':len(rows),**{k:sum(bool(r[k]) for r in rows) for k in ('ci','safe_success','refused')},'invalid':sum(r.get('invalid',False) for r in rows)};summaries.append(summary);print(json.dumps(summary),flush=True)
        del model;import gc;gc.collect()
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'summaries':summaries,'scope':'native output fields, synthetic initial knowledge/norm; not natural-language CI validation'},indent=2))

if __name__=='__main__':run(sys.argv[1])
