"""Check source-role/time consistency of visible receipts vs source document claims."""
from pathlib import Path
from collections import Counter
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import PUBLIC_TASKS,PublicCIEnv
out=[]
count=Counter()
for s in PUBLIC_TASKS:
    env=PublicCIEnv(s,"linked",False,"minimal")
    st=env.initial()
    texts=[env.clues.get(m["content"].get("fact_id"),"")
           for m in st.native.ledger if m["recipient"]=="R"]
    same_request=env.spec.user_request
    flagged=[x for x in texts if "current document" in x.lower() or
             "current service document" in x.lower()]
    print(s.id,"PRIOR",texts,"CURRENT_MARKED_IN_PRIOR",len(flagged))
    count["histories"]+=1
    count["inconsistent_current_in_prior"]+=bool(flagged)
    out.append({"task":s.id,"prior_messages":texts,
                "prior_receipts_described_as_current":len(flagged)})
data={"counts":dict(count),"rows":out,
      "note":"A literal 'current' in a prior delivered receipt can be reasonable if describing a document that was current at an earlier time, but no timestamp/disambiguation is present here; subjective chronology review needed."}
p=ROOT/"artifacts"/"20261119_composition_signals"/"public_workflow_ci_10_final"/"FIRST_PASS_TEMPORAL_LANGUAGE_AUDIT.json"
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
