"""V4 train/dev coverage and role-distribution audit with DG36 fully excluded."""
from pathlib import Path
from collections import Counter,defaultdict
import json
ROOT=Path(__file__).resolve().parents[1]
V4=Path(r"D:\research\kci_data_v4_pilot_20261122\artifacts\20261122_kci_data_v4_pilot_v1")
OUT=ROOT/"artifacts/20261123_v4_release_control_v1"
def read(name):
    return [json.loads(s) for s in (OUT/name).open(encoding="utf8") if s.strip()]
def main():
    roots={}
    for raw in (V4/"ROOTS.jsonl").open(encoding="utf8"):
        if '"DG36' in raw or '"final_test"' in raw:continue
        x=json.loads(raw)
        assert x["root_id"] in ("DG31","DG32","DG33","DG34","DG35")
        roots[x["root_id"]]={
          "root":x["root_id"],"split":x["split"],"business":x["business_family"],
          "graph_mechanism":x["graph_mechanism"],"pivot":x["actual_decision_pivot"],
          "public_norm_human_review":x["human_real_world_privacy_norm_validation"],
          "n_native_rules":len(x["native_actual_rules"])}
    assert len(roots)==5
    train=read("TRAIN_RELEASE_GROUPS.jsonl");dev=read("DEV_NATIVE_STATES_DG34_DG35.jsonl")
    assert len(train)==9 and len(dev)==10
    by_root=defaultdict(lambda:{"train_release_comparisons":0,"train_release_categories":[],"train_roles":set(),
                                "dev_roles":set(),"dev_conditions":Counter()})
    for r in train:
        b=by_root[r["root"]];b["train_release_comparisons"]+=1
        b["train_release_categories"].append(r["category"]);b["train_roles"].add(r["role"])
    for r in dev:
        b=by_root[r["root"]];b["dev_roles"].add(r["role"])
        b["dev_conditions"][r["world"].rsplit("__",1)[1]]+=1
    audit=[]
    for root in sorted(roots):
        b=by_root[root]
        audit.append({**roots[root],
           "train_release_independent_comparisons":b["train_release_comparisons"],
           "train_categories":sorted(b["train_release_categories"]),
           "roles_train":sorted(b["train_roles"]), "roles_dev":sorted(b["dev_roles"]),
           "dev_conditions":dict(b["dev_conditions"])})
    out={
       "status":"DEV_GENERALIZATION_COVERAGE_AUDIT_ONLY",
       "DG36_test_never_deserialized":True,
       "n_train_roots":3,"n_dev_roots":2,
       "n_train_independent_release_comparisons":len(train),
       "n_dev_native_release_states":len(dev),
       "dev_roles_unseen_in_train":sorted({r["role"] for r in dev}-{r["role"] for r in train}),
       "dev_graph_mechanisms_unseen_in_train":sorted({roots[x["root"]]["graph_mechanism"] for x in dev}-
                 {roots[x["root"]]["graph_mechanism"] for x in train}),
       "root_table":audit}
    file=OUT/"COVERAGE_AUDIT.json"
    assert not file.exists()
    file.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(out,ensure_ascii=False,indent=2)[:6500])
if __name__=="__main__":main()
