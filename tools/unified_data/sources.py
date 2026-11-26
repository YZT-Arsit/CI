"""Typed source admissions and explicit split constraints."""
from collections import defaultdict
import json
import native_rows as n

p = n.legacy
cn = n.cn
ART, V9, CORE = n.ART, p.V9, n.CORE
PHASE = ART/"20261124_cce_downstream_sft_v1"
DG = ART/"20261122_kci_R_task_witness_train_v1"
API = p.BASE.parent/"kci_api_output"
SELECT = ART/"20261125_process_policy"
TASK = CORE/"BUSINESS_NORM_SUPPLEMENT_V1"
V4 = p.BASE.parent/"kci_data_v4_phase2_20261122/artifacts"
V4_FILES = [
    ("20261122_kci_data_v4_pilot_v1","TRAIN_SAME_STATE_PREFERENCES.jsonl","FILE_HASHES_SHA256.json"),
    ("20261122_kci_data_v4_phase2_full_v1","TRAIN_METHOD_COMPAT.jsonl","FILE_SHA256.json"),
    ("20261122_kci_v4_grounded_language_v1","TRAIN_SAME_STATE_NARRATIVE_DPO.jsonl","FILE_SHA256.json")]


def typed(rs):
    for r in rs:
        _, ob, label = p.pack(r)
        kind = label["kind"]
        tier = r["tier"]
        group = "task" if tier in ("TASK_ONLY","train_task_readiness","train_task_terminal") else "ci"
        route = {"completion":"task_sft","pair":"task_dpo" if group=="task" else "dpo",
                 "masked":"task_sft" if group=="task" else "sft","multilabel":"static_evidence","hold":"controls"}[kind]
        if kind=="completion":
            positives=[label["target"]]
        elif kind=="multilabel":
            assert set(label["pos"]+label["neg"])==set(ob["actions"])
            positives=[cn({"selected_entry_ids":sorted(r["supervision"]["positive_entry_ids"])})]
            ob=n.observation(ob["ctx"],[])
        else:
            positives=label["pos"]
        meta={k:r[k] for k in ("stage","stratum","value_tier","core_mechanism","evidence_grade","norm_evidence_grade",
                                "actual_model_failure_origin","source_license","external_release_status","scope") if k in r}
        yield n.example(r,ob,positives,tier,meta,negatives=[] if kind=="multilabel" else label["neg"],route=route,
                        grade=r.get("evidence_tier",r.get("evidence_grade",kind)))


def supplementary(core):
    for directory,name,_ in V4_FILES:
        for x in n.lines(V4/directory/name):
            assert x["split"]=="train"
            menu=x.get("all_executable_actions",x.get("option_order"))
            pre,*opts=x["prompt"].rsplit("\n",len(menu))
            pre=pre.rsplit("\n",1)[0]
            labels=[z.split(". ",1) for z in opts]
            assert [k[0] for k in labels]==[chr(65+i) for i in range(len(menu))]
            a,b=x["chosen_action"],x["rejected_action"]
            assert a==menu[ord(x.get("chosen_letter") or x["chosen"])-65] and b==menu[ord(x.get("rejected_letter") or x["rejected"])-65]
            root=x.get("root",x.get("root_id"));dom="PHASE2" if root.startswith("P") else "DG"
            root=dom+":"+root
            ctx=dict(actor=x.get("actor") or x.get("role") or x["acting_role"],role_visible_text=pre,
                     action_descriptions=dict(zip(menu,[v for _,v in labels])))
            r=dict(source_group_key=root,group=root,source_dataset=dom,row_id=x.get("preference_id") or x["semantic_input_hash"])
            yield n.example(r,n.observation(ctx,menu),[cn(a)],directory,dict(category=x.get("category",x.get("branch")),
                            surface=x.get("surface"),language=x.get("language")),negatives=[cn(b)],route="dpo",grade=x["evidence_grade"])
    for x in n.lines(PHASE/"SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl"):
        assert x["split"]=="train" and x["verified_native_full_success_witness"] and x["role"]==x["native_role_observation"]["actor"]
        root="PHASE2:"+x["root"]
        r=dict(source_group_key=root,group=root,source_dataset="PHASE2",row_id=x["same_observation_sha256"])
        pp=list(x["multiple_successful_native_action_witnesses"])
        assert x["chosen_action"] in pp
        yield n.example(r,n.observation(x["native_role_observation"],x["all_executable_actions"]),list(map(cn,pp)),
                        "full_team_task_witness",dict(witness=x["native_path_witness_hash"]),
                        route="task_sft",grade="native_safe_full_team_witness")
    for x in n.lines(DG/"DPO_R_TASK_AND_CI_TRAIN.jsonl"):
        assert x["split"]=="train"
        root="DG:"+x["root"]
        menu=x["all_executable_actions"]
        text,options=x["prompt"].rsplit("Executable actions (reply with one option letter):",1)
        mapping = [line.split(". ",1) for line in options.strip().splitlines()]
        assert [z[0] for z in mapping]==[chr(65+i) for i in range(len(menu))]
        assert menu[ord(x["chosen_letter"])-65]==x["chosen_action"] and menu[ord(x["rejected_letter"])-65]==x["rejected_action"]
        ctx=dict(actor=x["native_role"],role_visible_text=text.strip(),action_descriptions=dict(zip(menu,[t for _,t in mapping])))
        r=dict(source_group_key=root,group=root,source_dataset="DG",row_id=p.key([x["state_hash"],x["chosen_action"],x["rejected_action"]]))
        task=x["source"]=="verified_native_completion_vs_safe_abort"
        yield n.example(r,n.observation(ctx,menu),[cn(x["chosen_action"])],"native_DG_"+x["source"],
                        dict(world=x["world"],state=x["state_hash"]),negatives=[cn(x["rejected_action"])],
                        route="task_dpo" if task else "dpo",grade=x["evidence_type"])
    for x in n.lines(API/"REVIEWED_TRAIN_CANDIDATES.jsonl"):
        m=x["metadata"]; assert m["split"]=="train" and m["certified"] and m["natural_language_semantics_verified"]
        ctx=json.loads(x["prompt"]);a,b=map(json.loads,(x["chosen"],x["rejected"]))
        assert a["actor"]==b["actor"]==ctx["role"]
        assert set(json.loads(ctx["task"])["actions"])=={a["kind"],b["kind"]}
        root="MAP-FILE:"+next(iter(m["source_hashes"].values()))
        r=dict(source_group_key=root,group=root,source_dataset="MAP_ADAPTED_QUERY",row_id=m["language_variant_id"])
        yield n.example(r,n.observation(ctx,[a,b]),[cn(a)],"reviewed_finite_action_language_variant",
                        dict(parent=m["language_parent_record_id"],group=m["base_case_group_id"]),negatives=[cn(b)],
                        route="dpo",grade="exact_finite_adapted_protocol")
    obs=p.rgz(p.OLD/"observations.gz")
    demo=p.rgz(SELECT/"TRAIN_DEMOS.gz")
    for g in p.rgz(p.OLD/"graphs.gz"):
        if g["partition"]!="train":
            continue
        tags={t for w in g["worlds"].values() for t in [w["head"]]+[b["tag"] for b in w["paths"].values()]}
        for t in sorted(tags & demo.keys()):
            ob=obs[t]
            ctx=json.loads(ob["prompt"].split("\n",1)[1].split("\nExecutable actions:\n")[0])
            for v in demo[t]:
                assert v["action"] in ob["actions"]
                # Action stays bound to its own private record; no gold record in the input.
                completion=cn({"private_record":v["record"],"action":json.loads(v["action"])})
                yield n.example(core[g["row_id"]],dict(ctx=ctx,actions=sorted(ob["actions"])),[completion],
                                "verified_private_process_demo",dict(native_action=v["action"]),route="process_sft",
                                grade="teacher_process_action")


class Groups:
    def __init__(self):
        self.parent={}
    def find(self,x):
        self.parent.setdefault(x,x)
        while x!=self.parent[x]:
            self.parent[x]=self.parent[self.parent[x]]
            x=self.parent[x]
        return x
    def join(self,x,y):
        x,y=sorted([self.find(x),self.find(y)])
        self.parent[y]=x


def split_registry(rs):
    groups=Groups(); roles=defaultdict(set); records=[]; pins={}
    def load(path):
        pins[str(path)]=p.sha(path)
        return p.rd(path)
    def mark(s,k,origin):
        roles[s].add(k);records.append(dict(source=s,status=k,origin=origin))
    core={r["row_id"]:r for r in rs if r["tier"]=="core_reference"}
    for r in rs:
        groups.join(r["source_group_key"],r["group"])
        if r["source_group_key"].startswith("TOP-R:"):
            groups.join(r["group"],r["source_group_key"].split("__")[0])
    for s in n.lines(n.data/"PROOFWRITER_CORE_PREREG_V1/PREREG_MINED_SPECS.jsonl"):
        groups.join(s["source_group_key"],"PW-source:"+s["original_source_id"])
        groups.join(s["source_group_key"],"PW-theory:"+str(s["theory_signature"]))
    for s in n.lines(n.data/"NEGOTIATION_CORE_INDEPENDENT_V3/SOURCE_ALIASES.jsonl"):
        if s["canonical_id"] in core:
            groups.join("DEAL_OR_NO_DEAL:deal-"+s["source_root"],core[s["canonical_id"]]["group"])
    for s in n.lines(n.data/"AGGREGATE_K_CORE_V1/PREREG_ALL_CANDIDATES.jsonl"):
        groups.join(s["source_group_key"],"MAP-FILE:"+s["source_sha256"])
    for s in n.lines(n.dynamic.aud.REL/"data/PREREG_RELATIONAL_TASKS_V2/PREREG_TRAIN_MECHANISMS_WITH_QUERIES.jsonl"):
        groups.join("MULTIAGENTPRIVACY:"+s["case_root_id"],"MAP-FILE:"+s["source_sha256"])
    for d,parts in load(V9/"validation/SOURCE_SPLITS.json").items():
        for k,roots in parts.items():
            if k!="train":
                for r in roots:mark(d+":"+r,k,"V9 original source split")
    prospective=V9/"validation/PROSPECTIVE_24_ORIGINAL_SOURCE_METADATA_ONLY.jsonl"
    pins[str(prospective)]=p.sha(prospective)
    for r in n.lines(prospective):
        mark("PISAS:"+r["source_group"],"prospective","unexecuted metadata")
    for k in ("P211","P212"):mark("PHASE2:"+k,"locked_test","historical seal")
    mark("DG:DG36","locked_test","historical seal")
    mark("PHASE2:P207","dev","diagnostic")
    for file,field in [("20261124_v9_training/freeze.json","groups"),("20261124_v9_history_repair/freeze.json","groups"),
                       ("20261125_icml_v02/ROOT_MANIFEST.json","roots"),("20261125_icml_v02/B1/ROOT_MANIFEST.json","roots"),
                       ("20261125_continuation_4b/SPLIT.json","roots"),("20261125_knowledge_joint_4b/binding.json","selected"),
                       ("20261125_multisource_policy/SOURCE_SPLITS.json","rows"),("20261125_readout_generalization/fresh_freeze.json","selected")]:
        for row in load(ART/file)[field]:
            partition=row.get("partition",row.get("split","dev"))
            for s in ("source_root","root","split_group","source"):
                if row.get(s):
                    mark(row[s],"train_used" if partition in ("train","fit") else "dev",file)
            rid=row.get("id",row.get("row",row.get("row_id")))
            if rid in core:
                mark(core[rid]["group"],"train_used" if partition in ("train","fit") else "dev",file)
    for r in n.lines(PHASE/"SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl"):roles["PHASE2:"+r["root"]].add("train_used")
    for r in n.lines(DG/"DPO_R_TASK_AND_CI_TRAIN.jsonl"):roles["DG:"+r["root"]].add("train_used")
    for directory,name,_ in V4_FILES:
        for r in n.lines(V4/directory/name):
            root=r.get("root",r.get("root_id"))
            roles[("DG:" if root.startswith("DG") else "PHASE2:")+root].add("train_used")
    flags=defaultdict(set)
    for group,states in roles.items():flags[groups.find(group)].update(states)
    return groups,flags,records,pins

