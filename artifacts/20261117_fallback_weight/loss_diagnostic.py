"""Post-result TRAIN diagnostic: optimal-set probability versus greedy action."""
import gc
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path[:0]=[str(R.parent/'20261117_frozen_crossover'),str(R.parent/'20261117_teacher_binding')]
from common import verify,load,torch
from binding_loss import forward_targets,answer_ids
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
rows=lambda p:list(map(json.loads,p.read_text().splitlines()))


def main():
    plan=read(R/'LOSS_DIAGNOSTIC_PLAN.json');assert not (R/'loss_probabilities.jsonl').exists()
    assert sha(Path(__file__))==plan['source_sha256'] and sha(R/'PLAN.json')==plan['main_plan_sha256']
    main_plan=read(R/'PLAN.json');seal=read(R/'SEALED.json')
    assert seal['complete'] and all(sha(R/n)==h for n,h in seal['files'].items())
    assert all(sha(ROOT/n)==h for n,h in main_plan['dependencies'].items());verify()
    observations=rows(R.parent/'20261117_option_fallback/observations.jsonl')[:108]
    labels=rows(R.parent/'20261117_option_fallback/labels.jsonl')[:108]
    assert len(observations)==108 and all(o['group']=='zero' for o in observations)
    results=[];durations=[]
    for arm in main_plan['arms']:
        cp=R/'repair'/arm;cpplan={'path':cp.relative_to(ROOT).as_posix(),'seal':read(cp/'CHECKPOINT_SEALED.json')}
        tokenizer,model=load('arm',{'checkpoints':{'arm':cpplan}});letters=answer_ids(tokenizer);stats=Counter()
        generated=rows(cp/'train_orders.jsonl')[:108]
        assert [o['id'] for o in observations]==[r['id'] for r in generated]
        for start in range(0,108,4):
            batch=observations[start:start+4];t=time.perf_counter()
            with torch.inference_mode():
                logits,_,_=forward_targets(tokenizer,model,[o['prompt'] for o in batch],['ANSWER:\nA']*len(batch))
                logp=torch.log_softmax(logits[:,-2],-1);probs=logp[:,letters].exp().tolist();top=logp.argmax(-1).tolist()
            seconds=time.perf_counter()-t;assert seconds<12;durations.append(seconds)
            for i,(obs,ps,token_id) in enumerate(zip(batch,probs,top)):
                label=labels[start+i];assert obs['id']==label['id']
                good=[j for j,a in enumerate(obs['actions']) if a in label['optimal_actions']]
                bad=[j for j in range(3) if j not in good];assert len(good)==2 and len(bad)==1
                selected=obs['actions'][letters.tolist().index(token_id)] if token_id in letters.tolist() else None
                old=generated[start+i];matches=re.fullmatch(r'ANSWER:\n([ABC])\s*',old['raw'])
                old_selected=obs['actions'][ord(matches[1])-65] if matches else None
                optimal=selected in label['optimal_actions'];goodmass=sum(ps[j] for j in good)
                union_beats_bad=goodmass>max(ps[j] for j in bad)
                record={'arm':arm,'id':obs['id'],'probabilities_ABC':ps,'other_probability':max(0.,1-sum(ps)),
                    'greedy_token_id':token_id,'greedy_action':selected,'optimal':optimal,'optimal_union_probability':goodmass,
                    'best_optimal_probability':max(ps[j] for j in good),'best_nonoptimal_probability':max(ps[j] for j in bad),
                    'union_beats_bad':union_beats_bad,'free_output_prefix_matches':matches is not None,
                    'free_action_agrees':matches is not None and old_selected==selected,'seconds_batch':seconds}
                with (R/'loss_probabilities.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                stats.update(n=1,optimal=int(optimal),unsafe_greedy=int(selected is not None and not optimal),
                    union_wins_but_greedy_unsafe=int(union_beats_bad and selected is not None and not optimal),
                    qualified_union_wins_but_greedy_unsafe=int(union_beats_bad and selected is not None and not optimal and matches is not None and old_selected==selected),
                    other_argmax=int(selected is None),
                    prefix_matches=int(matches is not None),free_agrees=int(matches is not None and old_selected==selected))
            del logits,logp
        results.append({'arm':arm,**dict(stats),'all_cases_interpretation_eligible':stats['prefix_matches']==stats['free_agrees']==108});del model;gc.collect()
    report={'scope':'TRAIN oracle-input post-result loss diagnostic, not new generalization or causal training proof',
            'summaries':results,'forwards':len(durations),'seconds':sum(durations),'max_seconds':max(durations),
            'model_updates':0,'probabilities_sha256':sha(R/'loss_probabilities.jsonl'),'whole_goal_complete':False}
    (R/'LOSS_DIAGNOSTIC_RESULTS.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
