"""Isolated CPU dependency path, all tests split into <=9.5s test-module jobs."""
from lm_common import *
import subprocess,time

def run():
    receipts=[];out=R/'regressions';out.mkdir(exist_ok=True)
    envvars=os.environ.copy();envvars['PYTHONPATH']=os.pathsep.join([str(ROOT/'code'),env['lm_dependency_path'],env['torch_path']])
    for file in sorted((ROOT/'code/tests').glob('test_*.py')):
        t=time.perf_counter()
        try:
            p=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',file.name],cwd=ROOT/'code',env=envvars,capture_output=True,text=True,timeout=9.5)
            record={'module':file.name,'returncode':p.returncode,'seconds':time.perf_counter()-t,'status':'passed' if p.returncode==0 else 'failed'}
            (out/(file.stem+'.log')).write_text(p.stdout+p.stderr)
        except subprocess.TimeoutExpired as e:record={'module':file.name,'status':'timeout','seconds':time.perf_counter()-t}
        receipts.append(record);print(json.dumps(record),flush=True)
    (out/'RESULTS.json').write_text(json.dumps(receipts,indent=2))

if __name__=='__main__':run()
