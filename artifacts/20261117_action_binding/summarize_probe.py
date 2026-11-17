"""Read sealed logits only; never equate an oracle prefix with autonomous success."""
import json,sys,hashlib,math
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))
from kci.action_binding_metrics import pair_metrics

def run():
    seal=json.loads((R/'PROBE_SEALED.json').read_text(encoding='utf-8'));summaries=[];effects=[]
    for model in seal['models']:
        path=R/(model['model']+'.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==model['outputs_sha256']
        records=list(map(json.loads,path.read_text(encoding='utf-8').splitlines()));assert len(records)==144
        for split in ('train','new_dev'):
            for mode in ('direct','gold_trace','swapped_trace'):
                rs=[r for r in records if r['split']==split and r['mode']==mode]
                for depth in (None,1,2,3):
                    group=[r for r in rs if depth is None or r['depth']==depth]
                    selected=pair_metrics(group);legal=pair_metrics([{**r,'selected':r['legal_argmax_action']} for r in group])
                    bits={k:[b for r in group for b in r['semantic_bits'] if b['kind']==k] for k in ('SAFE','TASK')}
                    summaries.append({'model':model['model'],'split':split,'mode':mode,'depth':depth,'states':len(group),'full_vocab_pairs':selected,'legal_argmax_pairs':legal,'full_vocab_state_correct':sum(r['selected']==r['optimal_action'] for r in group),'mean_legal_mass':sum(r['legal_mass'] for r in group)/len(group),'teacher_forced_bits':{k:{'n':len(v),'full_vocab_correct':sum(b['full_vocab_correct'] for b in v)} for k,v in bits.items()}})
            indexed={(r['pair_id'],r['side'],r['mode']):r for r in records if r['split']==split}
            keys=sorted({(r['pair_id'],r['side']) for r in records if r['split']==split});changes=[]
            for pair_id,side in keys:
                def target_p(mode):
                    r=indexed[pair_id,side,mode];return r['legal_probabilities'][r['actions'].index(r['optimal_action'])]/r['legal_mass']
                changes.append(target_p('gold_trace')-target_p('swapped_trace'))
            effects.append({'model':model['model'],'split':split,'mean_target_probability_gold_minus_swapped':sum(changes)/len(changes),'states':len(changes),'positive_above_1e_4':sum(x>1e-4 for x in changes),'scope':'supplied teacher-evidence sensitivity, not learned relation extraction'})
    result={'summaries':summaries,'prefix_effects':effects,'autonomous_goal_passed':False,'limitations':['forced prefixes are diagnostic, not free generation','gold and swapped traces contain oracle labels','TRAIN means original training-pool worlds rerendered with new names/order, not verified previously sampled prompts','seed17 only for bottleneck localization; no three-seed repair evidence','SAFE labels at these Planner states are all positive; TASK bits carry the relational distinction']}
    (R/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    for x in summaries:
        if x['depth'] is None:print(json.dumps(x))
    print(json.dumps({'prefix_effects':effects}))

if __name__=='__main__':run()
