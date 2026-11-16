"""Stop only this run's exact obsolete evaluator command; preserve partial files."""
import os,json,signal
from pathlib import Path
R=Path(__file__).resolve().parent
target=['python3','-B',str(R/'evaluate_policies.py')]
matches=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
    except (OSError,UnicodeDecodeError):continue
    if args==target:matches.append(int(p.name))
assert len(matches)<=1
for pid in matches:os.kill(pid,signal.SIGTERM)
(R/'slow_evaluation_stop.json').write_text(json.dumps({'exact_command':target,'terminated_owned_pids':matches,'reason':'validated source-input cache eliminates repeated mounted-filesystem loads; partial results retained and not inspected'},indent=2))
print(json.dumps({'owned_processes_stopped':len(matches)}))
