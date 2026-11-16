"""Exact expected action exposure under both direct-learning samplers."""
import json,collections
from pathlib import Path
R=Path(__file__).resolve().parent

def run():
    rows=[json.loads(l) for p in sorted((R/'dataset/train').glob('states_*.jsonl')) for l in p.read_text().splitlines()];items=[];old={};paired={};metadata={}
    for row in rows:
        o=row['observation']
        for c in row['candidates']:
            i=len(items);stratum=(o['role'],c['action'].split(':')[0],o['contract']['permission'],o['contract']['binding_required'],c['immediate_safe'],c['safe_completion']);items.append(stratum)
            old.setdefault((o['role'],c['immediate_safe'],c['safe_completion']),[]).append(i)
            key=(row['base_group_id'],o['role'],o['committed_ticket'],c['action']);paired.setdefault(key,[]).append(i);metadata[key]=stratum[:4]
    new={}
    for key,ids in paired.items():new.setdefault((*metadata[key],*items[ids[0]][-2:],*items[ids[1]][-2:]),[]).append(ids)
    probabilities={name:collections.defaultdict(float) for name in ('independent','paired_factors')}
    for ids in old.values():
        for i in ids:probabilities['independent'][items[i]]+=1/len(old)/len(ids)
    for pairs in new.values():
        for ids in pairs:
            for i in ids:probabilities['paired_factors'][items[i]]+=1/len(new)/len(pairs)/2
    result={'independent_buckets':len(old),'paired_factor_buckets':len(new),'strata':[{'stratum':list(k),'independent_probability':probabilities['independent'][k],'paired_probability':probabilities['paired_factors'][k]} for k in sorted(set(items))]}
    assert all(abs(sum(p.values())-1)<1e-8 for p in probabilities.values())
    (R/'SAMPLING_AUDIT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([r for r in result['strata'] if r['stratum']==['Worker','PUBLISH_FULL',False,True,1,1]]))

if __name__=='__main__':run()
