"""Verify role-visible solvability and actual checkpoint exposure for the content contrasts."""
import json
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

from check import OLD, OUT, S1, matched, read

sys.path.insert(0, str(OLD.parents[2] / "kci_supervision_alignment_20261123/d_future_core_20261124/tools"))
from probe_proofwriter_core import atom, ground, inv


def main():
    graphs, obs = read(OLD/"graphs.gz"), read(OLD/"observations.gz")
    logs = [json.loads(l) for l in (S1/"updates.jsonl").read_text().splitlines()]
    exposures=Counter((u["root"],tag,v) for u in logs for tag in u["tags"] for v in u["views"])
    stats = {}
    @lru_cache(None)
    def solve(tag):
        row=obs[tag]
        c=json.loads(row["prompt"].split("\n",1)[1].split("\nExecutable actions:\n")[0])
        facts=[r["literal"] for r in c["recipient_actual_deliveries"]]
        rules=[dict(premises=r["if"],conclusion=r["then"],id=i) for i,r in enumerate(c["public_rule_logic"])]
        prot=atom(c["fixed_norm"]["protected_proposition"]["literal"])
        forbidden={prot,inv(prot)}
        q1={atom(r["literal"]) for r in c["first_slot_eligible_claims"]}
        q2=atom(c["second_slot_required_claim"]["literal"])
        allowed=set();depths={}; stopped=[]
        for key in row["actions"]:
            a=json.loads(key)
            if a["kind"]=="SAFE_ABORT":
                stopped.append(key);continue
            init,rs=ground(dict(facts=facts+[a["literal"]],rules=rules))
            k=set(init);d=0;hit=None
            while True:
                if hit is None and forbidden & k:hit=d
                new={v for prem,v,_ in rs if prem<=k}-k
                if not new:break
                k.update(new);d+=1
            depths[key]=hit
            if hit is None and q1 & k and q2 in k:allowed.add(key)
        return allowed or set(stopped),depths
    for g in graphs:
        root=g["root"]; count=stats.setdefault(g["partition"],Counter())
        for a,b,tail in matched(g):
            ps=g["worlds"]["original"]["paths"];tags=[ps[x]["tag"] for x in (a,b)]
            exposed=min(exposures[root,t,v] for t in tags for v in (0,1))
            count[f"pairs_min_exposures_per_history_and_order_{exposed}"]+=1
            for tag,x in zip(tags,(a,b)):
                accept,dep=solve(tag)
                assert accept==set(ps[x]["accepted"])
                count["full_menu_accept_set_matches"]+=1
                if x==b:
                    assert dep[tail] is not None
                    count[f"conflict_proof_rounds_{dep[tail]}"]+=1
                else:assert dep[tail] is None
    (OUT/"VISIBLE_AND_EXPOSURE.json").write_text(json.dumps(stats,indent=2),encoding="utf8")
    print(stats)


if __name__=="__main__":main()
