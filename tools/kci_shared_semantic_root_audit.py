"""Audit per-root and per-graph results of shared semantic-action policy."""
from pathlib import Path
from collections import Counter,defaultdict
import json,argparse,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci.diverse_ci_branch_env import task_catalog
TASK={t.id:t for t in task_catalog()}
def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--decisions",type=Path,required=True)
    p.add_argument("--training",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    assert not args.output.exists()
    data=rows(args.decisions)
    assert data
    group=defaultdict(list)
    for r in data:
        assert r["position_invariance_exact"]
        assert r["root"] in TASK
        group[(r["dataset"],r["root"])].append(r)
    per_root=[]
    for (dataset,root),items in sorted(group.items()):
        pairs=defaultdict(list)
        for x in items:
            stem=x["world"] if x["dataset"].startswith("release") else x["world"].split("__")[0]
            pairs[stem].append(x)
        twin=[v for v in pairs.values() if len(v)==2]
        per_root.append({
           "dataset":dataset,"split":items[0]["split"],"root":root,"graph_family":TASK[root].graph,
           "business":TASK[root].business,"n_states":len(items),
           "correct":sum(x["correct"] for x in items),
           "paired_worlds":len(twin),"both_correct_pairs":sum(all(x["correct"] for x in v) for v in twin),
           "predicted_counts":dict(Counter(x["predicted_native_action"] for x in items)),
           "truth_counts":dict(Counter(x["truth"] for x in items))})
    group_graph=defaultdict(list)
    for r in per_root:
        group_graph[(r["dataset"],r["split"],r["graph_family"])].append(r)
    per_graph=[{
      "dataset":ds,"split":s,"graph_family":g,
      "business_roots":len(rs),
      "states_correct":sum(x["correct"] for x in rs),
      "states_n":sum(x["n_states"] for x in rs),
      "both_correct_pairs":sum(x["both_correct_pairs"] for x in rs),
      "paired_worlds":sum(x["paired_worlds"] for x in rs),
    } for (ds,s,g),rs in sorted(group_graph.items())]
    train=None
    if args.training:
        train=json.loads(args.training.read_text(encoding="utf8"))
    o={"n_decision_states":len(data),
       "n_independent_business_roots":len({r["root"] for r in per_root}),
       "n_distinct_structural_graphs":len({r["graph_family"] for r in per_root}),
       "root_table":per_root,"graph_table":per_graph,
       "shared_training_sampled_roots_by_group":train.get("sampled_roots_by_group") if train else None,
       "shared_training_sampled_roles_by_group":train.get("sampled_roles_by_group") if train else None,
       "root_split_not_blind":True}
    args.output.write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("PER_ROOT_AUDIT",json.dumps(per_graph,ensure_ascii=False)[:13000])
if __name__=="__main__":main()
