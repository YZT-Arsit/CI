"""Finish read-only validation and report assembly after all writers seal outputs."""
import subprocess,sys,time,json
from pathlib import Path
R=Path(__file__).resolve().parent
required=['language_native_transfer/EVALUATION_SEALED.json','balanced_language_forced_states/EVALUATION_SEALED.json','time_budget_probe/RESULTS.json','execution_audits/balanced_language_evaluation_SUMMARY.json']
while not all((R/p).exists() for p in required):time.sleep(2)
for script,args in [('audit_native.py',['language']),('audit_forced.py',[]),('verify_final_run.py',[]),('summarize.py',[]),('write_report.py',[])]:
    with (R/(script[:-3]+'_final.log')).open('w',encoding='utf-8') as log:
        p=subprocess.run([sys.executable,'-B',str(R/script),*args],stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'script':script,'returncode':p.returncode}),flush=True)
    if p.returncode:sys.exit(p.returncode)
