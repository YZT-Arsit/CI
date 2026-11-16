"""Stage selected evidence and refresh the repository's byte-level upload manifest."""
import json,hashlib,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]

def run():
    required=['REPORT.md','compression_manifest.json','RESULTS.json']
    if not all((R/p).exists() for p in required):raise ValueError('final report/archive not ready')
    selected=[p.relative_to(ROOT).as_posix() for p in R.rglob('*') if p.is_file() and p.suffix not in ('.jsonl','.pyc') and '__pycache__' not in p.parts and p.stat().st_size]
    selected += ['README.md','code/experiments/relation_solution_data.py','code/kci/relation_solution.py','code/kci/relation_language.py','code/kci/relation_runtime.py','code/kci/extended_meeting_tasks.py','code/tests/test_relation_solution.py','tools/restore_relation_artifacts.py']
    pathfile=ROOT/'.upload_work/relation_pathspec.bin';pathfile.parent.mkdir(exist_ok=True);pathfile.write_bytes(b'\0'.join(p.encode() for p in sorted(set(selected)))+b'\0')
    subprocess.run(['git','add','-f','--pathspec-from-file='+str(pathfile),'--pathspec-file-nul'],cwd=ROOT,check=True)
    tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0');records=[]
    for name in sorted(p for p in tracked if p and p!='UPLOAD_MANIFEST.json'):
        p=ROOT/name;raw=p.read_bytes();records.append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
    manifest={'snapshot_date':'2026-11-16','repository':'YZT-Arsit/CI','scope':'Research evidence through relation/safety-utility repair: validated structured reference, all nine pretrained CPU LoRA checkpoints and negative results, preserved historical runs. GPU large-scale NO_GO. JSONL restores losslessly.',
        'excluded':['paper PDFs','large pretrained LLM base weights and installed dependencies','raw JSONL represented by verified gzip','empty worker logs','most older unselected intermediate runs'],'files':records}
    (ROOT/'UPLOAD_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');subprocess.run(['git','add','UPLOAD_MANIFEST.json'],cwd=ROOT,check=True)
    print(json.dumps({'files':len(records),'bytes':sum(r['bytes'] for r in records),'new_selected':len(selected)}))

if __name__=='__main__':run()
