import sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'balanced_language_evaluation/EVALUATION_SEALED.json').exists():time.sleep(2)
for script,args in [('evaluate_balanced_forced_states.py',['balanced_language']),('audit_executions.py',['balanced_language_evaluation'])]:
    with (R/(script.replace('.py','')+'_balanced.log')).open('w') as log:
        subprocess.run([sys.executable,'-B',str(R/script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
