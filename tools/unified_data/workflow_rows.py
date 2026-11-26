"""Recover reference actions at relay, delegate and executor states from native TRAIN episodes."""
from collections import Counter
import native_rows as n

D = n.dynamic
SCOPE = D.aud.SCOPE/"data/RECIPIENT_SCOPE_EXACT_TRAIN_V1/EVAL_ONLY_COMPLETE_NATIVE_EPISODES.jsonl.gz"
TEMP = D.aud.TEMP/"data/TEMPORAL_FULL_BUSINESS_TRAIN_V1/EVAL_ONLY_FULL_BUSINESS_BRANCHES.jsonl.gz"


def workflow_rows(unused):
    inv,_,_,specs,pin = D.load_data()
    mapping = {}
    for r in inv.values():
        if r["stage"] not in ("PISAS_SCOPE_SAFE_DELIVERY_QMU","PISAS_TEMPORAL_FULL_BUSINESS_QMU"):
            continue
        src=D.original(r["source_file"])[r["source_line_zero_based"]]
        mapping[src["mechanism_id"]]=r
    for family,archive in [("scope",SCOPE),("temporal",TEMP)]:
        assert n.legacy.sha(archive)==n.legacy.rd(archive.parent/"FILE_SHA256.json")[archive.name]
        seen=set()
        for x in n.lines(archive):
            if not x["weight"] or not x.get("safe_full_success",x.get("outcome",{}).get("safe_success")):
                continue
            mid=x["mechanism_id"]
            spec=specs[mid]
            r=mapping.get(mid)
            if r is None:
                # Native TRAIN sources with tied first actions still have valid downstream witnesses.
                assert spec["split"]=="train"
                w=D.aud.load_world(D.aud.P/"dataset/sources/pisas",spec["source_record"],pin)
                env=D.scope_env.ScopeEnv(w,spec) if family=="scope" else D.temporal_env.Env(w,spec)
                group="PISAS:"+w.scenario_group_id
                r=dict(row_id=mid,source_dataset="PISAS",source_group_key=group)
            else:
                env,_=D.instance(family,r["row_id"])
            r=dict(r,group=r["source_group_key"])
            is_scope=family=="scope"
            sid=env.state_id if is_scope else env.sid
            st=env.initial() if is_scope else env.after_first(x["first_action"],x["availability"])
            if not is_scope:assert sid(st)==x["after_reveal_id"]
            buf=[]
            for step in x.get("outcome",x)["trace"]:
                assert sid(st)==step["before"]
                if "action" not in step:
                    assert is_scope
                    st,_=env.reveal(st,x["availability"])
                else:
                    a=step["action"]
                    ctx=env.observation(st)
                    menu=ctx.pop("candidate_actions")
                    assert a in menu and step["actor"]==ctx.get("actor",ctx.get("acting_agent"))
                    if step["before"] not in seen:
                        buf.append((step["before"],n.observation(ctx,menu),a))
                    st=env.step(st,a)[0] if is_scope else env.step(st,a)
                assert sid(st)==step["after"]
            assert st.done and not st.violation and (st.success if is_scope else st.full_success)
            for state,ob,a in buf:
                # Distinct chances can produce identical states; preserve distinct actions there.
                k=state,n.cn(a)
                if k in seen:continue
                seen.add(k)
                yield n.example(r,ob,[n.cn(a)],family+"_downstream_witness",
                    dict(mechanism=mid,state=state,terminal=sid(st),chance=x["availability"],weight=x["weight"],denominator=x["denominator"]),
                    grade="executed_safe_success_witness")

