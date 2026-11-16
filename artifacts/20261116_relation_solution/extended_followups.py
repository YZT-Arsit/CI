"""Resume previously deferred diagnostics after the user's explicit time extension."""
import sys,time,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;kind=sys.argv[1]
wait_for='language_external_complete' if kind=='language' else 'balanced_language_evaluation'
while not (R/wait_for/'EVALUATION_SEALED.json').exists():time.sleep(2)
script='evaluate_forced_states.py' if kind=='language' else 'evaluate_balanced_forced_states.py'
with (R/(kind+'_forced_extended.log')).open('w') as log:
    subprocess.run([sys.executable,'-B',str(R/script),kind],stdout=log,stderr=subprocess.STDOUT,check=True)
