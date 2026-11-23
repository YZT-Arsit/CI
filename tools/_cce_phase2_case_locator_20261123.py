"""Find user-named Phase-2 case IDs in local project source, no mutations."""
from pathlib import Path
import os,re
ROOT=Path(r"D:\research\longhistory")
NEED=("P201","P203","P205","P208")
ok={".md",".py",".json",".jsonl",".txt",".csv",".yaml",".yml"}
results=[]
for sub in ("docs","tools","code","artifacts"):
    for dr,ds,fs in os.walk(ROOT/sub):
        ds[:]=[x for x in ds if x not in (".git","__pycache__",".venv","node_modules") and not x.startswith("pytest")]
        for name in fs:
            p=Path(dr)/name
            if p.suffix.lower() not in ok:continue
            try:
                size=p.stat().st_size
                if size>3_000_000:continue
                data=p.read_text(encoding="utf8",errors="ignore")
            except (OSError,UnicodeError):continue
            hits=[x for x in NEED if x.lower() in data.lower()]
            if not hits:continue
            lns=data.splitlines()
            samples=[(i+1,s[:260]) for i,s in enumerate(lns) if any(x in s.upper() for x in hits)][:3]
            results.append((str(p.relative_to(ROOT)),size,",".join(hits),samples))
for row in sorted(results,key=lambda r:(0 if r[0].startswith("docs") else 1,0 if r[0].startswith("code") else 1,len(r[0])))[:170]:
    print(repr(row),flush=True)
print("TOTAL_MATCHING_FILES",len(results),flush=True)
