"""Verify equal token lengths, vocabularies and action order for matched histories."""
from lm_common import *
from kci.relation_language import vocabulary
from experiments.relation_solution_data import environment
from experiments.receipt_dataset import observations
import time,hashlib

def load_worlds(split):return [json.loads(l) for p in sorted((R/'dataset'/split).glob('worlds_*.jsonl')) for l in p.read_text().splitlines()]

def run():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(env['native_lm_path'],local_files_only=True);pools=palette(tokenizer);groups={}
    for w in load_worlds('train'):groups.setdefault(w['base_group_id'],[]).append(w)
    pairs=[tuple(sorted(g,key=lambda w:w['world_id'])) for g in groups.values()]
    for split in ('test','depth4','depth6','depth10','external_complete'):
        right={w['world_id']:w for w in load_worlds(split+'_swap')}
        pairs += [(w,right[w['world_id']]) for w in load_worlds(split)]
    counts={'world_pairs':len(pairs),'state_palette_pairs':0,'different_token_sequences':0,'unequal_lengths':0,'unequal_vocabularies':0,'unequal_action_orders':0};maximum=0;errors=[]
    for left,right in pairs:
        t=time.perf_counter();le=environment(left);re=environment(right)
        for index,(ls,rs) in enumerate(zip(observations(le),observations(re))):
            lo=le.observation(ls);ro=re.observation(rs);nonce=seed_for(left['base_group_id'],index)
            for name in ('train','fresh'):
                pool=pools[name];a,aa=prompt(lo,pool,nonce);b,ba=prompt(ro,pool,nonce);ai=tokenizer.encode(a);bi=tokenizer.encode(b)
                counts['state_palette_pairs']+=1;counts['different_token_sequences']+=ai!=bi;counts['unequal_lengths']+=len(ai)!=len(bi);counts['unequal_vocabularies']+=vocabulary(lo,pool,nonce)!=vocabulary(ro,pool,nonce);counts['unequal_action_orders']+=aa!=ba
                if len(ai)!=len(bi) or aa!=ba:errors.append([left['world_id'],index,name,len(ai),len(bi)])
        maximum=max(maximum,time.perf_counter()-t)
    result={**counts,'errors':errors,'max_pair_seconds':maximum,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (R/'PROMPT_PAIR_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert not errors and not counts['unequal_vocabularies']

if __name__=='__main__':run()
