"""Complete primary proof auditing; omit the cancelled supplemental forced states."""
import sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
while not (R/'balanced_language_evaluation/EVALUATION_SEALED.json').exists():time.sleep(2)
with (R/'audit_executions_balanced.log').open('w') as log:
    subprocess.run([sys.executable,'-B',str(R/'audit_executions.py'),'balanced_language_evaluation'],stdout=log,stderr=subprocess.STDOUT,check=True)
