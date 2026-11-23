"""Exact finite-state CPU checks: authorized prompt aliasing and optimal value conflict.
No model weights, GPU, or new RL experiment.
"""
from pathlib import Path
from dataclasses import asdict
from collections import deque,defaultdict,Counter
from functools import lru_cache
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.agent_rl.state_restarts import RestartSpec,restore
from kci.agent_rl.core import digest,prompt_for,terminal_reward
f=ROOT/"artifacts"/"20261119_composition_signals"/"agent_rl_v1"/"policy_qpi_01"/"selection_v2"/"FORKS.jsonl"
raw=[json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
prompts=defaultdict(list)
one_step=Counter()
from collections import defaultdict
for i,r in enumerate(raw):
    spec=RestartSpec.from_record(r)
    env,initial=restore(spec)
    @lru_cache(None)
    def v(state_key):
        state=state_cache[state_key]
        if state.done:
            return terminal_reward(success=state.success,violation=state.violation,done=state.done,steps=state.turn)
        q={}
        for a in env.options(state):
            nxt, audit=env.transition(state,a)
            assert audit["verdict"]!="UNKNOWN"
            k=digest(asdict(nxt))
            state_cache[k]=nxt
            q[a]=v(k)
        return max(q.values())
    state_cache={}
    queue=deque([initial])
    seen=set()
    while queue:
        s=queue.popleft()
        k=digest(asdict(s))
        if k in seen:continue
        seen.add(k);state_cache[k]=s
        if s.done:continue
        opts=env.options(s)
        text=prompt_for(env.observation(s),opts)
        vals={}
        for a in opts:
            t,audit=env.transition(s,a)
            if audit["verdict"]=="UNKNOWN":raise AssertionError("Unexpected unknown")
            tk=digest(asdict(t));state_cache[tk]=t
            vals[a]=v(tk)
            queue.append(t)
        ordered=sorted(vals.values(),reverse=True)
        margin=(ordered[0]-ordered[1]) if len(ordered)>1 else 0.
        best={a for a,w in vals.items() if ordered[0]-w<=1e-6}
        prom_key=hashlib.sha256(text.encode("utf-8")).hexdigest()
        prompts[prom_key].append({"scenario":spec.scenario.key,"pair":spec.pair_id,
                                 "state_hash":k,"role":env.actor(s),
                                 "options":list(opts),"qstar":vals,
                                 "best":sorted(best),"gap":margin,
                                 "related":spec.scenario.related})
        one_step["states"]+=1
conflicts=[]
for key,items in prompts.items():
    if len(items)<2:continue
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            a,b=items[i],items[j]
            if set(a["best"]).isdisjoint(b["best"]):
                conflicts.append({"prompt_hash":key,"a":a,"b":b,
                                  "material_gap":min(a["gap"],b["gap"])>=0.1})
print(json.dumps({
 "restarted_worlds":len(raw),
 "nonterminal_state_entries_with_world_context":one_step["states"],
 "unique_authorized_prompts":len(prompts),
 "prompts_repeated_across_enumerated_worlds":sum(len(x)>1 for x in prompts.values()),
 "entries_at_repeated_prompts":sum(len(x) for x in prompts.values() if len(x)>1),
 "conflicting_optimal_actions_for_identical_prompt_pairs":len(conflicts),
 "conflicting_optimal_action_pairs_with_both_margins_at_least_0.1":sum(c["material_gap"] for c in conflicts),
 "conflict_examples":conflicts[:3],
 "interpretation":"Qstar is used ONLY to test input identifiability, not for training. A prompt collision with conflicting optimal actions would be impossible to solve perfectly for a stateless policy. Absence of collision here does NOT prove the 4B model can compute optimal actions or generalize."
},ensure_ascii=False,indent=2),flush=True)
