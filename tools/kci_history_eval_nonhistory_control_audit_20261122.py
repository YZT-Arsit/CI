"""Audit whether opposite history examples accidentally differ in non-history prompt fields."""
from pathlib import Path
from collections import Counter,defaultdict
import json
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"
def nonhistory(prompt):
    return [line for line in prompt.splitlines() if not line.startswith("Delivered receipt ")]
def main():
    items=[json.loads(x) for x in (D/"PAIR_STATES.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    assert len(items)==56
    pairs=defaultdict(dict)
    for r in items:pairs[r["world"]][r["branch"]]=r
    rows=[]
    for world,by in pairs.items():
        assert set(by)=={"safe","risky"}
        a=nonhistory(by["risky"]["visible_prompt"])
        b=nonhistory(by["safe"]["visible_prompt"])
        diff=[{"risky":aa,"safe":bb} for aa,bb in zip(a,b) if aa!=bb]
        if len(a)!=len(b):
            diff.append({"length_diff":[len(a),len(b)]})
        rows.append({"world":world,"root":by["risky"]["root"],
                     "split":by["risky"]["split"],"source_mode":by["risky"]["source_mode"],
                     "strict_two_way":by["risky"]["aware_label"]=="VERIFY_PRIVATE_R" and by["safe"]["aware_label"]=="SEND_RAW_R",
                     "nonhistory_identical":not diff,"differences":diff})
    selected=[r for r in rows if r["strict_two_way"]]
    report={
      "n_worlds":len(rows),"n_two_way_worlds":len(selected),
      "strict_two_way_nonhistory_identical":sum(r["nonhistory_identical"] for r in selected),
      "strict_two_way_identical_by_source_mode":{
           mode:sum(r["nonhistory_identical"] for r in selected if r["source_mode"]==mode)
           for mode in sorted({r["source_mode"] for r in selected})},
      "nonhistory_differing_fields":dict(Counter(x["risky"].split(":",1)[0] for r in selected for x in r["differences"] if "risky" in x)),
      "note":"All prior delivered receipt content removed; compare remaining exact prompt lines.",
      "per_world":rows}
    out=D/"NONHISTORY_MATCH_AUDIT.json"
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({k:v for k,v in report.items() if k!="per_world"},ensure_ascii=False,indent=2))
    for r in selected:
        if r["differences"]:
            print("DIFFERENCE_EXAMPLE",r["world"],json.dumps(r["differences"][:3],ensure_ascii=False)[:1500])
            break
if __name__=="__main__":main()
