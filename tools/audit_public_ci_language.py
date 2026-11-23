"""Review-list diagnostics for unapproved API paraphrases, CPU-only.
No script ever modifies the native oracle or marks semantic equivalence approved.
"""
from pathlib import Path
import json,re,sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
sys.path.insert(0,str(ROOT/"tools"))
from kci.public_workflow_ci import PUBLIC_TASKS,SEEDS,graph_texts
from run_public_ci_baselines import extract_edges,text_derived_private
D=ROOT/"artifacts"/"20261119_composition_signals"/"public_workflow_ci_10_final"
specs={s.id:s for s in PUBLIC_TASKS}
p=D/"API_LANGUAGE_REVIEW_REQUIRED.jsonl"
rows=[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
assert len(rows)==10 and len(set(r["task_id"] for r in rows))==10
record=[]
counts=Counter()
for r in rows:
    s=specs[r["task_id"]]
    orig=r["four_original_source_messages"]
    cand=r["four_candidate_source_messages"]
    assert len(orig)==len(cand)==4
    assert orig==[graph_texts(s,SEEDS[s.seed_id])[f] for f in ("f0","f1","f2","f3")]
    flags=[]
    if len(cand)!=4:flags.append("bad_count")
    for i,(o,n) in enumerate(zip(orig,cand)):
        assert isinstance(n,str) and n.strip()
        orig_ids=set(re.findall(r"\b[WR]\d\d-[A-Z]\d+\b",o))
        new_ids=set(re.findall(r"\b[WR]\d\d-[A-Z]\d+\b",n))
        if orig_ids!=new_ids:flags.append(f"reference_mismatch_{i}")
        if "Case owner Rowan" in n and "restricted category" in n.lower():
            flags.append(f"single_fragment_direct_sensitive_link_{i}")
        if re.search(r"\b(no longer|not associated|not linked|different from|incorrect|fake|unknown)\b",n,re.I):
            flags.append(f"negation_or_uncertainty_{i}")
    before=sum(len(extract_edges(x)) for x in orig)
    after=sum(len(extract_edges(x)) for x in cand)
    related_orig=[x for x in orig if ("earlier" in x.lower() or "prior" in x.lower())]
    counts["total"]+=1
    counts["reference_preserved"]+=int(not any("reference_mismatch" in x for x in flags))
    counts["flagged_roots"]+=int(bool(flags))
    counts["template_lexical_parser_complete"]+=int(before==after)
    counts["source_clause_count"]+=before
    counts["candidate_clause_count"]+=after
    record.append({
        "root_id":s.id,"source_clauses_lexically_parsed":before,
        "candidate_clauses_lexically_parsed":after,
        "preliminary_structural_flags":flags,
        "manual_entailment_review":"PENDING",
        "semantic_certification":False,
    })
out=D/"API_LANGUAGE_DIAGNOSTIC.json"
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps({
    "status":"STRUCTURAL_AUDIT_ONLY",
    "counts":dict(counts),"rows":record,
    "note":"Template regex coverage is not a semantic equivalence test. Lower coverage can mean more diverse wording OR a changed entailment.",
    "autopromote_to_training":False,"source_truth_modified":False
},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(dict(counts),ensure_ascii=False,indent=2))
