"""Resume only the WSL regression step after the preserved 9.5s startup timeout."""
import json,hashlib,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OUT=ROOT/'.publish_verify/r16'

def run():
    tree='23ba0da2be329aca5a2ba13836d8499d40f76f19'
    assert subprocess.check_output(['git','write-tree'],cwd=ROOT,text=True).strip()==tree
    records=json.loads((OUT/'UPLOAD_MANIFEST.json').read_text(encoding='utf-8'))['files']
    data=json.loads((OUT/'artifacts/20261116_relation_solution/DATA_AUDIT.json').read_text(encoding='utf-8'))
    assert data['independent_target_check_passed'] and data['train_groups_disjoint'] and not data['errors']
    original=json.loads((R/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    assert data['files']==original['files'] and data['splits']==original['splits']
    def code_check():
        for row in records:
            if row['path'].startswith(('code/','tools/')):assert hashlib.sha256((OUT/row['path']).read_bytes()).hexdigest()==row['sha256']
    code_check()
    initial={'verified_tree_before_adding_this_receipt':tree,'manifest_files_verified':len(records),'initial_attempt':'verify_publish.py completed all manifest checks, both restores and independent data audit, then raised TimeoutExpired at the WSL test subprocess','initial_timeout_seconds':9.5,'original_predictions_changed':False}
    (R/'PORTABLE_INITIAL_TIMEOUT.json').write_text(json.dumps(initial,indent=2),encoding='utf-8')
    e=json.loads((R/'environment.json').read_text(encoding='utf-8'));unix='/mnt/'+OUT.drive[0].lower()+str(OUT)[2:].replace('\\','/')
    path=':'.join([unix+'/code',e['lm_dependency_path'],e['torch_path']]);t=time.perf_counter()
    try:
        p=subprocess.run(['wsl.exe','-e','env','PYTHONPATH='+path,'python3','-B','-m','unittest','discover','-s',unix+'/code/tests','-p','test_relation_solution.py'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=30)
        result={**initial,'resumed_test_returncode':p.returncode,'seconds_including_WSL_startup':time.perf_counter()-t,'output':p.stdout+p.stderr,'wrapper_timeout_seconds':30}
    except subprocess.TimeoutExpired as error:
        (R/'PORTABLE_RESUME_TIMEOUT.json').write_text(json.dumps({**initial,'second_timeout_seconds':30,'error':str(error)},indent=2),encoding='utf-8');raise
    code_check();result.update(archive_code_unchanged=True,verification_receipt_is_added_after_archive=True)
    (R/'PORTABLE_VERIFICATION.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result));assert p.returncode==0 and 'Ran 7 tests' in result['output'] and 'skipped=' not in result['output']

if __name__=='__main__':run()
