"""Build native teacher bindings once per independently reproducible source."""
import argparse
import hashlib
import json
import gzip
from pathlib import Path
import native_rows as n
import workflow_rows as workflow


def cache_inputs(name):
    a, d, table = n.ART, n.data, n.proof.TABLE
    paths = {
        "proofwriter":[table/"CASES.jsonl",table/"EVAL_ONLY_CELLS.jsonl",Path(n.proof.__file__),Path(n.proof.pw.__file__),n.proof.pw.P,d/"PROOFWRITER_CORE_PREREG_V1/PREREG_MINED_SPECS.jsonl"],
        "negotiation":[d/"NEGOTIATION_CORE_V2/EVAL_ONLY_ALL_NATIVE_BRANCHES.jsonl.gz",Path(n.native.ng.__file__),a.parent/"code/kci/v9_execution.py"],
        "aggregate":[d/"AGGREGATE_K_CORE_V1/PREREG_ALL_CANDIDATES.jsonl",Path(n.native.ag.__file__),Path(n.native.__file__)],
        "teacher_trajectories":[d/"EXISTING_602_CORE_AUDIT_V1/EVAL_ONLY_REPLAYED_CORE_AND_ABLATION.jsonl.gz",Path(n.dynamic.__file__),Path(n.dynamic.aud.__file__)],
        "historical_repairs":[n.legacy.OLD/"observations.gz",n.legacy.OLD/"graphs.gz",a/"20261125_process_policy/S3_SELECTION.json"],
        "workflow_rows":[workflow.SCOPE,workflow.TEMP,Path(workflow.__file__),Path(n.dynamic.temporal_env.__file__),Path(n.dynamic.scope_env.__file__)]
    }[name]
    return paths+[Path(n.__file__),n.CORE/"D_future_core_multisource_v2/D_future_core.jsonl"]


NAMES = ("proofwriter","negotiation","aggregate","teacher_trajectories","historical_repairs","workflow_rows")
CACHE = n.ROOT/".cache/unified_roles_v1"


def build(name):
    CACHE.mkdir(parents=True,exist_ok=True)
    out = CACHE/(name+".jsonl.gz")
    pins = {str(p):n.legacy.sha(p) for p in cache_inputs(name)}
    marker = CACHE/(name+".json")
    if marker.exists():
        old = n.legacy.rd(marker)
        if old["inputs"]==pins and out.exists() and old["sha256"]==n.legacy.sha(out):
            return old
    allrows,_,_ = n.legacy.inputs()
    rows = {r["row_id"]:r for r in allrows if r["tier"]=="core_reference"}
    tmp = out.with_suffix(".partial")
    count = 0
    with gzip.open(tmp,"wt",compresslevel=1,encoding="utf8") as file:
        generator = workflow.workflow_rows if name=="workflow_rows" else getattr(n,name)
        for r in generator(rows):
            file.write(n.cn(r)+"\n")
            count += 1
    tmp.replace(out)
    result = dict(rows=count,inputs=pins,sha256=n.legacy.sha(out))
    n.legacy.dump(marker,result)
    print(name,count,flush=True)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("source",choices=NAMES)
    build(ap.parse_args().source)

