"""Shared pinned CPU LM setup, formatting and finite candidate scoring."""
import sys,os,json,random
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
env=json.loads((R/'environment.json').read_text());sys.path[:0]=[env['lm_dependency_path'],env['torch_path'],str(ROOT/'code')]
os.environ['TOKENIZERS_PARALLELISM']='false'
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig,get_peft_model
from kci.relation_language import prompt,seed_for
torch.set_num_threads(4);torch.set_num_interop_threads(1)

def setup(seed):
    torch.manual_seed(seed)
    tokenizer=AutoTokenizer.from_pretrained(env['native_lm_path'],local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token;tokenizer.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(env['native_lm_path'],torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    return tokenizer,model

def palette(tokenizer):
    path=R/'entity_palettes.json'
    if not path.exists():
        words=[]
        for token in tokenizer.get_vocab():
            word=token.replace('\u0120','')
            if word.isascii() and word.isalpha() and word.islower() and 4<=len(word)<=6 and len(tokenizer.encode(' '+word,add_special_tokens=False))==1:words.append(word)
        words=sorted(set(words));random.Random(216031).shuffle(words)
        if len(words)<256:raise ValueError('insufficient single-token entity names')
        path.write_text(json.dumps({'train':words[:128],'fresh':words[128:256],'selection':'deterministic token inventory; no labels consulted'},indent=2))
    return json.loads(path.read_text())

def encode(tokenizer,texts,max_length=1024):
    rendered=[tokenizer.apply_chat_template([{'role':'user','content':t}],tokenize=False,add_generation_prompt=True) for t in texts]
    values=tokenizer(rendered,return_tensors='pt',padding=True,add_special_tokens=False)
    if values['input_ids'].shape[1]>max_length:raise ValueError('prompt exceeds explicit bound; no truncation')
    return values

def answer_ids(tokenizer):
    ids=[tokenizer.encode(letter,add_special_tokens=False) for letter in 'ABC']
    if any(len(x)!=1 for x in ids):raise ValueError('answer token boundary mismatch')
    return torch.tensor([x[0] for x in ids])

def load_states(split):
    return [json.loads(line) for p in sorted((R/'dataset'/split).glob('states_*.jsonl')) for line in p.read_text().splitlines()]

def model_logits(model,inputs):return model(**inputs,logits_to_keep=1,use_cache=False).logits[:,-1]
