"""Coverage-only follow-up: same frozen models, all eight fixed external sources."""
from evaluate_language import *
import evaluate_structured as structured

def run(kind):
    out=R/(kind+'_external_complete');out.mkdir(exist_ok=True)
    plan=out/'plan.json';assert not plan.exists()
    plan.write_text(json.dumps({'conditions':['external_complete','external_complete_swap'],'sources':list(range(10,18)),'coverage_repair':'native room-feature and calendar-deadline support; no training/model change','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    seal=R/(kind+'_models')/'CHECKPOINTS_SEALED.json'
    while not seal.exists():time.sleep(2)
    summaries=[]
    for report in json.loads(seal.read_text())['models']:
        folder=out/report['model'];folder.mkdir();arm=report['arm']
        if kind=='language':
            tokenizer,model=setup(report['seed']);adapter=R/'language_models'/report['model']/'adapter'
            assert all(hashlib.sha256((adapter/p).read_bytes()).hexdigest()==h for p,h in report['adapter_hashes'].items())
            set_peft_model_state_dict(model,load_file(str(adapter/'adapter_model.safetensors')));model.eval();names=palette(tokenizer)
        else:
            model=structured.DenseGraphPolicy() if arm=='dense_graph' else structured.LocalRulePolicy(no_binding=arm=='no_binding')
            path=R/'structured_models'/report['model']/'model.pt';assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_sha256']
            model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        for split in ('external_complete','external_complete_swap'):
            ws=worlds(split)
            if kind=='language':rows=evaluate(tokenizer,model,arm,ws,names['fresh'])
            else:
                ledger=[];rows=[structured.short(lambda:structured.execute(model,arm,w,811),ledger,'episode') for w in ws]
            (folder/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows),encoding='utf-8')
            summary={'model':report['model'],'condition':split,'episodes':len(rows),**{k:sum(r[k] for r in rows) for k in ('ci','safe_success','refused')}}
            summaries.append(summary);print(json.dumps(summary),flush=True)
        del model;import gc;gc.collect()
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'summaries':summaries},indent=2))

if __name__=='__main__':run(sys.argv[1])
