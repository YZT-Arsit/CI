"""Bounded CPU evaluation scheduling; never read or print policy results."""
import sys,time,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;pending=[2,3,4,5];running={};done={};start=time.monotonic()
while pending or running:
    for v,(p,log) in list(running.items()):
        if p.poll() is not None:
            done[v]={'exit':p.returncode,'sealed':(R/f'evaluation_v{v}_fast/EVALUATION_SEALED.json').exists()}
            log.close();del running[v]
            (R/'evaluation_scheduler_status.json').write_text(json.dumps(done,indent=2))
    for v in list(pending):
        if len(running)>=2:break
        if (R/f'models_v{v}/CHECKPOINTS_SEALED.json').exists():
            log=(R/f'evaluate_fast_v{v}_runner.log').open('w')
            running[v]=(subprocess.Popen([sys.executable,'-B',str(R/f'evaluate_fast_v{v}.py')],stdout=log,stderr=subprocess.STDOUT),log)
            pending.remove(v)
    if time.monotonic()-start>2100:raise TimeoutError('evaluation scheduling wall-clock limit')
    time.sleep(1)
print(json.dumps(done))
