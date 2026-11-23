"""Count the *actual full reward-bearing* native CI states, including violations.
Read-only CPU feasibility probe for exact frozen policy evaluation.
"""
from pathlib import Path
from collections import Counter,deque
from dataclasses import asdict
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.state_restarts import RestartSpec,restore
from kci.agent_rl.core import digest
P=ROOT/"artifacts"/"20261119_composition_signals"/"agent_rl_v1"/"policy_qpi_01"/"selection_v2"/"FORKS.jsonl"
raw=[json.loads(x) for x in P.read_text(encoding="utf-8").splitlines() if x.strip()]
total=set()
rows=[]
for i,r in enumerate(raw):
    spec=RestartSpec.from_record(r)
    env,initial=restore(spec)
    q=deque([initial])
    visited=set()
    terminal=Counter()
    transitions=0
    decisions=0
    while q:
        state=q.popleft()
        k=digest(asdict(state))
        if k in visited:continue
        visited.add(k)
        if state.done:
            label="violation" if state.violation else "success" if state.success else "incomplete"
            terminal[label]+=1
            continue
        decisions+=1
        for action in env.options(state):
            nxt, audit=env.transition(state,action)
            if audit["verdict"]=="UNKNOWN":
                raise AssertionError("Unknown native transition during exact-policy reachability")
            transitions+=1
            q.append(nxt)
    total.update(visited)
    rows.append({"index":i,"pair":spec.pair_id,"role":spec.role,"related":spec.scenario.related,
                 "states":len(visited),"nonterminal_states":decisions,"transitions":transitions,
                 "terminal":dict(terminal)})
print(json.dumps({"cases":len(rows),"sum_states":sum(r["states"] for r in rows),
 "max_states":max(r["states"] for r in rows),
 "sum_nonterminal_states":sum(r["nonterminal_states"] for r in rows),
 "max_nonterminal_states":max(r["nonterminal_states"] for r in rows),
 "unique_state_hashes_across_all":len(total),
 "rows":rows,"NOTE":"Includes all legal branches, violations and incomplete endings. Counting native states is CPU-only, not Qpi policy evaluation."},ensure_ascii=False,indent=2))
