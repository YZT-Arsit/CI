"""Frozen pretrained baseline on a declared subset, with adapters disabled."""
from evaluate_language import *

def run():
    out=R/'base_language_evaluation';out.mkdir(exist_ok=True);assert not (out/'plan.json').exists()
    splits=['test','test_swap','depth10','depth10_swap','optional','authorized','both','external_complete']
    (out/'plan.json').write_text(json.dumps({'conditions':splits,'worlds_per_split':16,'prompt':'same plain prompt','max_new_tokens':16,'adapters_disabled':True,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2))
    tokenizer,model=setup(17);model.eval();pool=palette(tokenizer)['fresh'];folder=out/'base';folder.mkdir();summaries=[]
    with model.disable_adapter():
        for split in splits:
            rows=evaluate(tokenizer,model,'plain',worlds(split),pool);(folder/(split+'.jsonl')).write_text(''.join(canonical(r)+'\n' for r in rows))
            summary={'model':'base','condition':split,'episodes':len(rows),**{k:sum(bool(r[k]) for r in rows) for k in ('ci','safe_success','invalid','refused')}};summaries.append(summary);print(json.dumps(summary),flush=True)
    (out/'EVALUATION_SEALED.json').write_text(json.dumps({'summaries':summaries,'format_control_limit':'same 16-token budget as plain; format failure is reported separately'},indent=2))

if __name__=='__main__':run()
