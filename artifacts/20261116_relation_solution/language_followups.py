"""Sequence supplemental runs after primary inference to avoid CPU contention."""
import sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'language_evaluation/EVALUATION_SEALED.json').exists():time.sleep(2)
for script,args in [('evaluate_external_complete.py',['language']),('evaluate_forced_states.py',['language']),('audit_executions.py',['language_evaluation']),('audit_executions.py',['language_external_complete'])]:
    with (R/(script.replace('.py','')+'_language.log')).open('w') as log:
        subprocess.run([sys.executable,'-B',str(R/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
