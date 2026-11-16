"""Disposable CPU LoRA timing probe; no pilot/evaluation examples used."""
import sys,os,json,time,threading,shutil
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
env=json.loads((R/'environment.json').read_text());sys.path[:0]=[env['lm_dependency_path'],env['torch_path']]
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from peft import LoraConfig,get_peft_model
torch.set_num_threads(4);torch.set_num_interop_threads(1)
source=json.loads((R/'lm_source_manifest.json').read_text())
cache=Path.home()/'.cache/kci_models_20261116'/source['revision'];cache.mkdir(parents=True,exist_ok=True)
for p in (ROOT/source['local_relative']).iterdir():
    target=cache/p.name
    if not target.exists():shutil.copy2(p,target)
env['native_lm_path']=str(cache);(R/'environment.json').write_text(json.dumps(env,indent=2))
start=time.perf_counter()
model=AutoModelForCausalLM.from_pretrained(cache,torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True)
model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0005)
results=[]
for length,batch in ((128,2),(256,2),(512,1)):
    def timeout():
        (R/'lm_profile_timeout.json').write_text(json.dumps({'length':length,'batch':batch}));os._exit(124)
    timer=threading.Timer(9.5,timeout);timer.daemon=True;timer.start();t=time.perf_counter()
    ids=torch.randint(10,1000,(batch,length));out=model(input_ids=ids,attention_mask=torch.ones_like(ids),logits_to_keep=1,use_cache=False)
    loss=torch.nn.functional.cross_entropy(out.logits[:,-1],torch.full((batch,),32,dtype=torch.long))
    loss.backward();opt.step();opt.zero_grad();timer.cancel()
    row={'length':length,'batch':batch,'seconds':time.perf_counter()-t,'finite_loss':bool(torch.isfinite(loss))};results.append(row);print(json.dumps(row),flush=True)
(R/'lm_profile.json').write_text(json.dumps({'cpu_only':True,'steps':results,'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),'setup_and_probe_seconds':time.perf_counter()-start},indent=2))
