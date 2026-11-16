"""Exploratory next-action stress of ALL frozen finite text-policy models.

No training, model selection, generated actions, or trajectory evaluation.
The word/adjacent-word hashed backbone is not an LLM context mechanism.
"""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiments.text_policy_probe import text_features,forward,action_class,KEYS,np
from kci.multi_env import FAMILIES

THRESHOLD=1e-9
VARIANTS=('sft_only','ci','future','all','balanced')
SEEDS=(17,29)


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines()]
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2),encoding='utf-8')
def save_lines(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')


def prepare(root,family):
    config=read(root/'plan.json');input_path=root/'input_snapshot.jsonl'
    assert all(digest(path)==expected for path,expected in config['source_sha256'].items())
    assert digest(input_path)==config['input_sha256']
    rows=[(i,r) for i,r in enumerate(lines(input_path)) if r['metadata']['rule_family_id']==family]
    assert len(rows)==40
    x=np.stack([text_features(row['prompt']) for _,row in rows]);meta=[]
    for index,row in rows:
        m=row['metadata'];spec=m['case_spec']
        meta.append({'input_index':index,'paired_group':m['matched_pair_group_id'],
            'family':family,'answer':spec['answer'],'count':spec['count'],'position':spec['position'],
            'relation':m['relation'],'prompt_sha256':sha256(row['prompt'].encode()).hexdigest(),
            'chosen_class':action_class(row['chosen']),'rejected_class':action_class(row['rejected']),
            'source_record_sha256':sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()})
    out=root/'features';out.mkdir(exist_ok=True)
    np.savez_compressed(out/f'{family}.npz',x=x)
    grouped=defaultdict(list)
    for i,m in enumerate(meta):grouped[m['paired_group']].append(i)
    distances=[]
    for group,indices in grouped.items():
        assert len(indices)==2 and {meta[i]['relation'] for i in indices}=={'related','unrelated'}
        distances.append({'paired_group':group,'count':meta[indices[0]]['count'],
            'position':meta[indices[0]]['position'],'feature_l2_distance':float(np.linalg.norm(x[indices[0]]-x[indices[1]]))})
    save(out/f'{family}.json',{'family':family,'rows':meta,'pair_feature_distances':distances,
        'features_sha256':digest(out/f'{family}.npz'),'matrix_bytes_sha256':sha256(x.tobytes()).hexdigest(),
        'shape':list(x.shape),'source_input_sha256':config['input_sha256'],
        'feature_source_sha256':config['source_sha256']['experiments/text_policy_probe.py']})
    print(json.dumps({'family':family,'rows':len(rows),'shape':list(x.shape)}))


def metrics(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['paired_group']].append(row)
    complete=[s for s in grouped.values() if len(s)==2 and {r['relation'] for r in s}=={'related','unrelated'}]
    return {'side_rows':len(rows),'correct_sides':sum(r['correct'] for r in rows),
        'side_accuracy':sum(r['correct'] for r in rows)/len(rows),
        'ties':sum(r['tie'] for r in rows),'complete_groups':len(complete),
        'joint_correct_groups':sum(all(r['correct'] for r in s) for s in complete),
        'joint_flip_rate':sum(all(r['correct'] for r in s) for s in complete)/len(complete) if complete else None,
        'chosen_margin_mean':float(np.mean([r['chosen_margin'] for r in rows])),
        'chosen_margin_change_from_zero_mean':float(np.mean([r['chosen_margin_change_from_zero'] for r in rows])),
        'chosen_margin_min':min(r['chosen_margin'] for r in rows),
        'chosen_margin_max':max(r['chosen_margin'] for r in rows),
        'by_relation':{relation:{'count':len(sub),'accuracy':sum(r['correct'] for r in sub)/len(sub),
            'chosen_margin_mean':float(np.mean([r['chosen_margin'] for r in sub])),
            'request_minus_raw_mean':float(np.mean([r['request_minus_raw'] for r in sub]))}
            for relation in ('related','unrelated') if (sub:=[r for r in rows if r['relation']==relation])}}


def evaluate(root,model_id):
    plan=read(root/'plan.json');model=next(m for m in plan['models'] if m['model_id']==model_id)
    assert all(digest(path)==expected for path,expected in plan['source_sha256'].items())
    weights=Path(model['weights_path']);assert digest(weights)==model['weights_sha256']
    assert digest(model['result_path'])==model['result_sha256']
    family=model['heldout_family'];meta=read(root/'features'/f'{family}.json')
    assert digest(root/'features'/f'{family}.npz')==meta['features_sha256']
    x=np.load(root/'features'/f'{family}.npz')['x'];values=np.load(weights)
    assert values['u'].shape==(24,512) and values['e'].shape==(len(KEYS),24)
    _,logits=forward(x,values['u'],values['e']);assert np.isfinite(logits).all()
    request=KEYS.index('B:REQUEST_RULES::');raw=KEYS.index('B:SEND:payload_0:');rows=[]
    for m,score in zip(meta['rows'],logits):
        chosen=float(score[m['chosen_class']]);rejected=float(score[m['rejected_class']]);margin=chosen-rejected
        rows.append({**m,'model_id':model_id,'variant':model['variant'],'seed':model['seed'],
            'chosen_logit':chosen,'rejected_logit':rejected,'chosen_margin':margin,
            'request_minus_raw':float(score[request]-score[raw]),'correct':margin>THRESHOLD,
            'tie':abs(margin)<=THRESHOLD,'all_action_logits':score.tolist()})
    baseline={(r['answer'],r['relation']):r for r in rows if r['count']==0}
    for row in rows:
        reference=baseline[row['answer'],row['relation']]
        row['zero_history_baseline_group']=reference['paired_group']
        row['chosen_margin_change_from_zero']=row['chosen_margin']-reference['chosen_margin']
    pair_rows=[]
    for group in dict.fromkeys(r['paired_group'] for r in rows):
        pair={r['relation']:r for r in rows if r['paired_group']==group};a=pair['related'];b=pair['unrelated']
        pair_rows.append({'paired_group':group,'model_id':model_id,'family':family,'variant':model['variant'],
            'seed':model['seed'],'count':a['count'],'position':a['position'],'answer':a['answer'],
            'related_chosen_margin':a['chosen_margin'],'unrelated_chosen_margin':b['chosen_margin'],
            'joint_flip_correct':a['correct'] and b['correct'],
            'semantic_request_margin_gap':a['request_minus_raw']-b['request_minus_raw']})
    pair_baseline={r['answer']:r for r in pair_rows if r['count']==0}
    for row in pair_rows:
        control=pair_baseline[row['answer']]
        row['zero_history_baseline_group']=control['paired_group']
        row['semantic_gap_change_from_zero']=row['semantic_request_margin_gap']-control['semantic_request_margin_gap']
    result={'model':model,'rows':rows,'paired_groups':pair_rows,'overall':metrics(rows),
        'by_count':{str(n):metrics([r for r in rows if r['count']==n]) for n in (0,8,64,256)},
        'by_count_position':{f'{n}:{p}':metrics(sub) for n in (0,8,64,256) for p in ('front','middle','end')
            if (sub:=[r for r in rows if r['count']==n and r['position']==p])},
        'feature_metadata_sha256':digest(root/'features'/f'{family}.json'),
        'frozen_weights_unchanged_after':digest(weights)==model['weights_sha256'],
        'scope':'next-action pair logits only, fixed finite word/bigram backbone, no model training or rollouts'}
    save(root/'models'/f'{model_id}.json',result)
    print(json.dumps({'model':model_id,'overall':result['overall']}))


def run(input_path,models_path,root):
    root.mkdir(parents=True,exist_ok=False);models=[]
    for path in sorted(models_path.glob('*/weights.npz')):
        result=read(path.parent/'result.json')
        assert digest(path)==result['weights_sha256']
        models.append({'model_id':path.parent.name,'heldout_family':result['heldout_family'],
            'variant':result['variant'],'seed':result['seed'],'weights_path':str(path.resolve()),
            'weights_sha256':digest(path),'result_path':str((path.parent/'result.json').resolve()),
            'result_sha256':digest(path.parent/'result.json')})
    expected={(f,v,s) for f in FAMILIES for v in VARIANTS for s in SEEDS}
    assert {(m['heldout_family'],m['variant'],m['seed']) for m in models}==expected and len(models)==40
    shutil.copy2(input_path,root/'input_snapshot.jsonl')
    source_paths=('experiments/text_history_stress.py','experiments/text_policy_probe.py')
    source_dir=root/'source';source_dir.mkdir()
    for source in source_paths:shutil.copy2(source,source_dir/Path(source).name)
    plan={'models':models,'input_path':str(input_path.resolve()),'input_sha256':digest(input_path),
        'hard_timeout_seconds':9.5,'correctness_margin_threshold':THRESHOLD,
        'families':list(FAMILIES),'variants':list(VARIANTS),'seeds':list(SEEDS),
        'source_sha256':{p:digest(p) for p in source_paths},'python_version':sys.version,'numpy_version':np.__version__,
        'mode':'posthoc exploratory stress, not used for model selection or retraining',
        'feature_semantics':'entire visible JSON prompt signed normalized hashed word/adjacent-word counts',
        'scope':'next-step pair logits only; finite backbone, not LLM long-context evaluation'}
    save(root/'plan.json',plan);receipts=[]
    for mode,identifier in [('prepare',f) for f in FAMILIES]+[('evaluate',m['model_id']) for m in models]:
        command=[sys.executable,'-B','-m','experiments.text_history_stress','--root',str(root.resolve()),'--worker',mode,'--identifier',identifier]
        started=time.perf_counter()
        try:
            process=subprocess.run(command,capture_output=True,text=True,timeout=9.5)
            receipt={'mode':mode,'identifier':identifier,'returncode':process.returncode,
                'seconds':time.perf_counter()-started,'hard_timeout_seconds':9.5,'stdout':process.stdout,'stderr':process.stderr}
        except subprocess.TimeoutExpired as error:
            receipt={'mode':mode,'identifier':identifier,'timeout':True,'seconds':time.perf_counter()-started,
                'hard_timeout_seconds':9.5,'stdout':str(error.stdout),'stderr':str(error.stderr)}
        receipts.append(receipt);save(root/'worker_receipts.json',receipts)
        if receipt.get('returncode')!=0:raise RuntimeError('preserve failed worker and stop')
    results=[read(root/'models'/f"{m['model_id']}.json") for m in models]
    rows=[r for result in results for r in result['rows']];pairs=[r for result in results for r in result['paired_groups']]
    save_lines(root/'scores.jsonl',rows);save_lines(root/'paired_scores.jsonl',pairs)
    # Group identities are repeated for each frozen model: prefix them only in
    # pooled aggregation so joint metrics never mix outcomes from different models.
    pooled=[{**r,'paired_group':r['model_id']+'|'+r['paired_group']} for r in rows]
    summary={'models':len(models),'feature_workers':len(FAMILIES),'successful_workers':len(receipts),
        'side_score_rows':len(rows),'paired_model_case_rows':len(pairs),
        'max_worker_seconds':max(r['seconds'] for r in receipts),'hard_timeout_seconds':9.5,
        'overall':metrics(pooled),'by_count':{str(n):metrics([r for r in pooled if r['count']==n]) for n in (0,8,64,256)},
        'by_variant':{v:metrics([r for r in pooled if r['variant']==v]) for v in VARIANTS},
        'model_summary':[{k:result['model'][k] for k in ('model_id','heldout_family','variant','seed')}|{'overall':result['overall'],'by_count':result['by_count']} for result in results],
        'frozen_weights_all_unchanged':all(result['frozen_weights_unchanged_after'] for result in results),
        'input_sha256':plan['input_sha256'],'source_sha256':plan['source_sha256'],
        'output_sha256':{n:digest(root/n) for n in ('scores.jsonl','paired_scores.jsonl')},
        'mode':plan['mode'],'scope':plan['scope'],'training_or_parameter_selection_performed':False,
        'actual_agent_rollouts_performed':False,'independent_task_generalization_established':False}
    save(root/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('models','side_score_rows','paired_model_case_rows','max_worker_seconds','overall','by_count')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=('prepare','evaluate'))
    parser.add_argument('--identifier');parser.add_argument('--input');parser.add_argument('--models');parser.add_argument('--root',required=True)
    args=parser.parse_args();root=Path(args.root)
    if args.worker=='prepare':prepare(root,args.identifier)
    elif args.worker=='evaluate':evaluate(root,args.identifier)
    else:run(Path(args.input),Path(args.models),root)
