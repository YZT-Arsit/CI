"""Small official-source downloads, one bounded subprocess per file; no credentials."""
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import Request,urlopen
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent
LIMIT=500_000

def worker(url,destination):
    if urlparse(url).hostname not in {'huggingface.co','api.github.com','raw.githubusercontent.com'}:
        raise ValueError('only declared official hosts')
    with urlopen(Request(url,headers={'User-Agent':'KCI-source-audit/1.0'}),timeout=7) as response:
        raw=response.read(LIMIT+1)
        if len(raw)>LIMIT:raise ValueError('single-file cap exceeded')
        destination.write_bytes(raw)
        return {'bytes':len(raw),'sha256':sha256(raw).hexdigest(),'final_url':response.url,
                'content_type':response.headers.get('Content-Type'),'etag':response.headers.get('ETag')}

if len(sys.argv)>1 and sys.argv[1]=='--worker':
    print(json.dumps(worker(sys.argv[2],Path(sys.argv[3]))))
else:
    url,relative=sys.argv[1:];destination=(ROOT/relative).resolve()
    if not destination.is_relative_to(ROOT) or destination.exists():raise ValueError('new artifact path required')
    source=relative.split('/')[0]
    used=sum(p.stat().st_size for p in (ROOT/source).rglob('*') if p.is_file()) if (ROOT/source).exists() else 0
    if used+LIMIT>5_000_000:raise ValueError('per-source cap')
    destination.parent.mkdir(parents=True,exist_ok=True)
    receipt={'url':url,'path':relative,'started_utc':datetime.now(timezone.utc).isoformat(),
             'hard_timeout_seconds':9.2,'single_file_byte_cap':LIMIT,'source_byte_cap':5_000_000}
    started=time.perf_counter()
    try:
        result=subprocess.run([sys.executable,'-B',str(Path(__file__)), '--worker',url,str(destination)],
                              timeout=9.2,capture_output=True,text=True)
        receipt.update(status='downloaded' if result.returncode==0 else 'error',returncode=result.returncode,
                       **(json.loads(result.stdout) if result.returncode==0 else {'error':result.stderr}))
    except subprocess.TimeoutExpired:receipt.update(status='timeout')
    receipt['worker_seconds']=time.perf_counter()-started
    with (ROOT/'download_receipts.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(receipt)+'\n')
    print(json.dumps(receipt))
