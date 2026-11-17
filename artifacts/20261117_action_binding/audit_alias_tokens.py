"""Check actual pinned-tokenizer lengths after response sealing, without a model."""
import json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'20261116_relation_solution'))
from lm_common import AutoTokenizer,env,encode

def main():
    out=R/'alias_confirmation';assert (out/'RESPONSES_SEALED.json').exists();tokenizer=AutoTokenizer.from_pretrained(env['native_lm_path'],local_files_only=True);tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left';details=[]
    for name in ['queries','actions']:
        rows=[json.loads(s) for s in (out/(name+'.jsonl')).read_text().splitlines()];inputs=encode(tokenizer,[r['prompt'] for r in rows]);lengths=inputs['attention_mask'].sum(1).tolist();groups={}
        for row,length in zip(rows,lengths):
            key=(row['group'],row['render'],row['action'] if name=='queries' else row['mode']);groups.setdefault(key,[]).append(length)
        assert all(len(v)==2 and v[0]==v[1] for v in groups.values())
        details.append({'kind':name,'prompts':len(rows),'equal_length_side_pairs':len(groups),'max_prompt_tokens':max(lengths),'min_prompt_tokens':min(lengths)})
    result={'pinned_model_revision':'12fd25f77366fa6b3b4b768ec3050bf629380bac','actual_chat_template_token_lengths':details,'scope':'exact paired prompt-length equality in this one-hop diagnostic; not long-context or scale proof','model_loaded':False};(out/'TOKEN_LENGTH_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
