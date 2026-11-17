"""Run the current suite in its existing native dependency environments."""
import json,os,sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]

def run():
    out=R/'regressions';out.mkdir(exist_ok=True);env=json.loads((ROOT/'artifacts/20261116_relation_solution/environment.json').read_text(encoding='utf-8'));variables=os.environ.copy();variables['PYTHONPATH']=os.pathsep.join([str(ROOT/'code'),env['lm_dependency_path'],env['torch_path']]);receipts=[]
    for file in sorted((ROOT/'code/tests').glob('test_*.py')):
        if file.name=='test_training_data.py':continue  # Existing Windows-only tokenizer binary.
        start=time.perf_counter()
        try:
            result=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p',file.name],cwd=ROOT/'code',env=variables,capture_output=True,text=True,timeout=9.5)
            (out/(file.stem+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8');receipt={'module':file.name,'returncode':result.returncode,'seconds':time.perf_counter()-start}
        except subprocess.TimeoutExpired:receipt={'module':file.name,'returncode':124,'seconds':time.perf_counter()-start}
        receipts.append(receipt);print(json.dumps(receipt),flush=True)
    (out/'WSL_RESULTS.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8');assert all(r['returncode']==0 for r in receipts)

if __name__=='__main__':run()
