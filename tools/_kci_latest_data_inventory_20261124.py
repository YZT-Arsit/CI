from pathlib import Path
from datetime import datetime
R=Path(r"D:\research")
def stamps(p,limit=45,dirs=True):
  if not p.exists():return
  xs=[]
  for z in p.iterdir():
    try:
      if not (z.is_dir() if dirs else z.is_file()):continue
      s=z.stat()
      xs.append((s.st_mtime,z.name, str(z),s.st_size))
    except OSError:continue
  for t,name,path,size in sorted(xs,reverse=True)[:limit]:
    print(datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"),name,"size",size,"path",path,flush=True)
for path,dirs,limit in [
 (R,True,50),
 (R/"longhistory"/"artifacts",True,35),
 (R/"longhistory"/"docs",False,25),
 (R/"kci_data_v4_phase2_20261122"/"artifacts",True,25),
 (R/"kci_task_pilot_20261122"/"artifacts",True,25),
 (R/"kci_v4_evidence_validation_20261122",True,25),
]:
 print("\n=== "+str(path)+" ===",flush=True)
 stamps(path,limit=limit,dirs=dirs)
