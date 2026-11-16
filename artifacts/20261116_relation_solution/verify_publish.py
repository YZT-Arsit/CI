"""Verify a fresh staged Git archive, restore data, and run the new tests in WSL."""
import json,hashlib,subprocess,zipfile,sys,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]

def run():
    tree=subprocess.check_output(['git','write-tree'],cwd=ROOT,text=True).strip();base=ROOT/'.publish_verify';base.mkdir(exist_ok=True);out=base/'r16';out.mkdir(exist_ok=False);archive=base/'r16.zip'
    subprocess.run(['git','archive','--format=zip','--output='+str(archive),tree],cwd=ROOT,check=True)
    with zipfile.ZipFile(archive) as z:
        if any(not (out/n).resolve().is_relative_to(out.resolve()) for n in z.namelist()):raise ValueError('archive path outside verification directory')
        z.extractall(out)
    records=json.loads((out/'UPLOAD_MANIFEST.json').read_text(encoding='utf-8'))['files']
    for row in records:
        p=out/row['path'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
    runs=[]
    for script in ('tools/restore_semantic_artifacts.py','tools/restore_relation_artifacts.py','artifacts/20261116_relation_solution/audit_data.py'):
        t=time.perf_counter();p=subprocess.run([sys.executable,'-B',str(out/script)],cwd=out,capture_output=True,text=True,encoding='utf-8',check=True);runs.append({'script':script,'returncode':p.returncode,'seconds':time.perf_counter()-t,'stdout_tail':p.stdout[-500:]})
    e=json.loads((R/'environment.json').read_text());unix='/mnt/'+out.drive[0].lower()+str(out)[2:].replace('\\','/')
    pythonpath=':'.join([unix+'/code',e['lm_dependency_path'],e['torch_path']]);t=time.perf_counter()
    # The wrapper includes cold WSL startup/imports, separately from unittest time.
    p=subprocess.run(['wsl.exe','-e','env','PYTHONPATH='+pythonpath,'python3','-B','-m','unittest','discover','-s',unix+'/code/tests','-p','test_relation_solution.py'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30)
    runs.append({'script':'archive WSL new regression tests','returncode':p.returncode,'seconds':time.perf_counter()-t,'output':p.stdout+p.stderr});assert p.returncode==0
    for row in records:
        if row['path'].startswith(('code/','tools/')):assert hashlib.sha256((out/row['path']).read_bytes()).hexdigest()==row['sha256']
    result={'verified_tree_before_adding_this_receipt':tree,'manifest_files_verified':len(records),'runs':runs,'archive_code_unchanged':True,'verification_receipt_is_added_after_archive':True}
    (R/'PORTABLE_VERIFICATION.json').write_text(json.dumps(result,indent=2));print(json.dumps({'manifest_files_verified':len(records),'new_regression_returncode':p.returncode,'restorations_passed':True}))

if __name__=='__main__':run()
