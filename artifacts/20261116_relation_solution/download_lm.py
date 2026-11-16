"""Pinned official pretrained weights; private workspace cache, not Git upload."""
import json,time,sys,os
from pathlib import Path
from hashlib import sha256
from urllib.request import urlopen,Request
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
repo='HuggingFaceTB/SmolLM2-135M-Instruct'
with urlopen('https://huggingface.co/api/models/'+repo,timeout=7) as response:meta=json.load(response)
revision=meta['sha'];out=ROOT/'.upload_work/models/SmolLM2-135M-Instruct'/revision;out.mkdir(parents=True,exist_ok=True)
(R/'lm_metadata.json').write_text(json.dumps(meta,indent=2))
names=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','model.safetensors']
available={r['rfilename'] for r in meta['siblings']};receipts=[]
for name in names:
    if name not in available:continue
    path=out/name;url=f'https://huggingface.co/{repo}/resolve/{revision}/{name}';start=time.perf_counter()
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.partial')
        with urlopen(Request(url,headers={'User-Agent':'KCI-CPU-research/1.0'}),timeout=15) as response,partial.open('wb') as f:
            while True:
                data=response.read(1024*1024)
                if not data:break
                f.write(data)
        partial.replace(path)
    receipt={'file':name,'url':url,'bytes':path.stat().st_size,'sha256':sha256(path.read_bytes()).hexdigest(),'seconds':time.perf_counter()-start}
    receipts.append(receipt);print(json.dumps(receipt),flush=True)
(R/'lm_source_manifest.json').write_text(json.dumps({'repository':repo,'revision':revision,'local_relative':out.relative_to(ROOT).as_posix(),'files':receipts},indent=2))
