"""Run a manifest of independent Python experiments with a hard per-job ceiling."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--timeout',type=float,default=9.5)
    args=p.parse_args()
    if not 0<args.timeout<=10:raise ValueError('per-job timeout must be at most 10 seconds')
    manifest=Path(args.manifest);jobs=json.loads(manifest.read_text(encoding='utf-8'))
    logs=manifest.parent/'job_logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env['PYTHONPATH']=str(Path.cwd());env['PYTHONDONTWRITEBYTECODE']='1'
    ledger=manifest.parent/(manifest.stem+'_receipts.jsonl')
    if ledger.exists():raise FileExistsError('Use a new manifest name; receipts are append-only')
    source_paths=sorted(p for folder in ('kci','experiments','tests') for p in Path(folder).glob('*.py'))
    source_hashes={str(p):sha256(p.read_bytes()).hexdigest() for p in source_paths}
    manifest.with_name(manifest.stem+'_source_manifest.json').write_text(
        json.dumps({'python':sys.version,'command_manifest_sha256':sha256(manifest.read_bytes()).hexdigest(),
                    'source_sha256':source_hashes},indent=2),encoding='utf-8')
    for job in jobs:
        command=[sys.executable,'-B',*job['args']]
        started=datetime.now(timezone.utc).isoformat();t=time.perf_counter()
        try:
            proc=subprocess.run(command,env=env,capture_output=True,text=True,encoding='utf-8',timeout=args.timeout)
            status='completed' if proc.returncode==0 else 'failed';rc=proc.returncode
            stdout=proc.stdout;stderr=proc.stderr
        except subprocess.TimeoutExpired as err:
            status='timeout';rc=None
            stdout=err.stdout or '';stderr=err.stderr or ''
            if isinstance(stdout,bytes):stdout=stdout.decode('utf-8',errors='replace')
            if isinstance(stderr,bytes):stderr=stderr.decode('utf-8',errors='replace')
        receipt={'id':job['id'],'started_utc':started,'seconds':time.perf_counter()-t,
                 'timeout_seconds':args.timeout,'status':status,'returncode':rc,'command':command}
        receipt['source_hashes_unchanged']=all(sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
        (logs/(job['id']+'.stdout.txt')).write_text(stdout,encoding='utf-8')
        (logs/(job['id']+'.stderr.txt')).write_text(stderr,encoding='utf-8')
        with ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(receipt)+'\n')
        print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
