"""Typed access to the V9 delivery and to the separately audited core reference pairs."""
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path

from .paper_worlds import canon, key


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    with Path(path).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    with Path(path).open("rb") as f:
        return sha256(f.read()).hexdigest()


def verify(folder):
    hashes=read(folder/"FILE_SHA256.json")
    for rel,digest in hashes.items():
        path=(folder/rel).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path)!=digest:
            raise ValueError(f"Integrity failure: {rel}")
    return len(hashes)


class Deliveries:
    def __init__(self,v9,core):
        self.core=Path(core).resolve()
        self.v9=Path(v9).resolve()
        source=self.v9.parents[1]/"validation_v9/tools/typed_training.py"
        spec=importlib.util.spec_from_file_location("v9_delivery_training",source)
        self.api=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.api)
        self.v9_files=verify(self.v9)
        self.typed=self.api.ValidatedCorpus(self.v9)
        self.core_files=verify(self.core)
        self.manifest=read(self.core/"MANIFEST.json")
        self.registry=read(self.core/"SOURCE_REGISTRY.json")

    def load_tier(self,tier,*,purpose,objective=None):
        return self.typed.get(tier,purpose=purpose,objective=objective)

    def core_reference(self,stratum,*,purpose):
        if purpose!="reference_distillation" or stratum not in self.manifest["strata"]:
            raise ValueError("Select one declared reference stratum for reference distillation")
        result=rows(self.core/(stratum+".jsonl"))
        for r in result:
            if (r["stratum"]!=stratum or r["split"]!="source_train" or r["reference_only_not_target_Qpi"] is not True
                or r["supervision"]["kind"]!="semantic_pair_dpo"):
                raise ValueError("Incompatible core target")
            good,bad=(r["supervision"][k] for k in ("chosen_action","rejected_action"))
            if good==bad or not all(a in r["executable_actions"] for a in (good,bad)):
                raise ValueError("Unbound semantic action")
            if key([r["actor_context"],sorted([good,bad],key=key)])!=r["pair_fingerprint"]:
                raise ValueError("Core context-action fingerprint mismatch")
        if len(result)!=self.manifest["strata"][stratum]:raise ValueError("Stratum count mismatch")
        return result

    def actor_prompt(self,context,actions,seed,view):
        menu=sorted(actions,key=canon)
        row=dict(actor_context=context,executable_actions=menu,supervision={"kind":"semantic_pair_dpo"},
                 input_semantic_sha256=key([context,menu]))
        return self.api.semantic_render(row,self.api.display_permutation(row,seed,view))

    def training_example(self,row,seed,view):
        if "stratum" in row:
            if row["stratum"] not in self.manifest["strata"] or not row["reference_only_not_target_Qpi"]:
                raise ValueError("Unexpected core supervision")
            return self.actor_prompt(row["actor_context"],row["executable_actions"],seed,view)
        return self.api.training_render(row,seed,view)

    def loss(self,row,*,purpose,seed,view,score,baseline=None):
        """Score a verified row. Semantic scorers take (prompt, canonical completions).

        For multilabel rows score takes (prompt, displayed evidence IDs) and returns
        one inclusion logit per ID. A DPO baseline is a frozen model likelihood.
        """
        import torch
        tier=row.get("evidence_tier","")
        if tier in ("HOLD_CONTROL_ONLY","REF_COST_OR_STOCHASTIC_DIAGNOSTIC"):
            raise ValueError("This tier is evaluator-only")
        expected="reference_distillation" if "stratum" in row or tier.startswith("REF_") else "training"
        if purpose!=expected:raise ValueError("Supervision purpose mismatch")
        text,actions=self.training_example(row,seed,view)
        kind=row["supervision"]["kind"]
        if kind=="multilabel_evidence_selection":
            order=self.api.display_permutation(row,seed,view)
            inventory=row["actor_context"]["evidence_available_from_tools"]
            ids=[inventory[i]["entry_id"] for i in order]
            return self.api.evidence_bce(score(text,ids),ids,row["supervision"])
        pos,neg=self.api.action_targets(row,actions)
        logps=score(text,actions)
        if kind=="masked_action_set":return self.api.masked_set_loss(logps,pos,neg)
        if kind!="semantic_pair_dpo" or baseline is None:raise ValueError("Semantic DPO requires frozen baseline")
        with torch.no_grad():frozen=baseline(text,actions).detach()
        return self.api.dpo_loss(logps,frozen,pos,neg)

    def audit(self):
        mf=self.typed.manifest
        tiers={}
        typed_ids=set()
        for tier,info in mf["tier_paths"].items():
            purpose="control" if tier=="HOLD_CONTROL_ONLY" else "diagnostics"
            rs=self.load_tier(tier,purpose=purpose)
            assert len(rs)==info["records"] and not typed_ids.intersection(r["row_id"] for r in rs)
            typed_ids.update(r["row_id"] for r in rs)
            for fmt in info["supervision_formats"]:
                use=(purpose if tier in ("HOLD_CONTROL_ONLY","REF_COST_OR_STOCHASTIC_DIAGNOSTIC")
                     else "reference_distillation" if tier.startswith("REF_") else "training")
                selected=self.load_tier(tier,purpose=use,objective=fmt)
                if use in ("training","reference_distillation"):
                    for r in selected:
                        _,opts=self.training_example(r,7,0)
                        if fmt!="multilabel_evidence_selection":self.api.action_targets(r,opts)
            tiers[tier]=dict(rows=len(rs),formats=Counter(r["supervision"]["kind"] for r in rs))
        inv=rows(self.v9/"inventory/ALL_2600_TYPED_RECORDS.jsonl")
        all_core=rows(self.core/"D_future_core.jsonl")
        index=rows(self.core/"EVALUATOR_ONLY_WITNESS_INDEX.jsonl")
        ids={r["row_id"]:r for r in all_core}
        assert len(ids)==len(all_core)==self.manifest["rows"]
        assert len(index)==len(ids) and {w["row_id"] for w in index}==ids.keys()
        sources={}
        for w in index:
            path=self.core.parent/w["source_file"].replace("\\","/")
            if path not in sources:
                digest=sha(path)
                assert digest in self.manifest["input_sha256"].values()
                sources[path]={r.get("row_id",r.get("id")):r for r in rows(path)}
            k=w["row_id"]; raw=sources[path][k]; r=ids[k]
            assert (w["source_root"],w["split_group"])==(r["source_group_key"],r["split_group"])
            assert raw["actor_context"]==r["actor_context"] and raw["executable_actions"]==r["executable_actions"]
            assert all(raw["supervision"][j]==r["supervision"][j] for j in ("chosen_action","rejected_action"))
        stratum_counts={}
        split_groups=set()
        for name in self.manifest["strata"]:
            rs=self.core_reference(name,purpose="reference_distillation")
            assert all(ids[r["row_id"]]==r for r in rs)
            for r in rs:
                text,options=self.training_example(r,7,0)
                positive,negative=self.api.action_targets(r,options)
                assert [options[i] for i in positive]==[canon(r["supervision"]["chosen_action"])]
                assert [options[i] for i in negative]==[canon(r["supervision"]["rejected_action"])]
            split_groups.update(r["split_group"] for r in rs)
            stratum_counts[name]=len(rs)
        assert sum(stratum_counts.values())==len(ids)
        for tier,f in (("SUCCESS","success"),("PROTECTION_COST","protection_cost")):
            rs=rows(self.core/("D_future_core_"+f+".jsonl"))
            assert rs==[r for r in all_core if r["value_tier"]==tier]
        prior=self.core.parent/"D_future_core"
        return dict(v9_files=self.v9_files,core_files=self.core_files,
            v9_build=mf["status"],v9_final=read(self.v9/"INDEPENDENT_AUDIT.json")["status"],
            v9_input=len(inv),v9_unique=len(typed_ids),v9_tiers=tiers,
            duplicates=mf["duplicate_records_quarantined"],v9_core_common_ids=len(typed_ids&ids.keys()),
            core_total=len(ids),strata=stratum_counts,core_value_tiers=Counter(r["value_tier"] for r in all_core),
            by_source=Counter(r["source_dataset"] for r in all_core),split_groups=len(split_groups),
            roots=len({r["source_group_key"] for r in all_core}),core_source_hashes={p.name:sha(p) for p in sources},
            existing_reference_rechecked=read(prior/"MANIFEST.json")["existing_inventory_audited"],
            existing_reference_retained=read(prior/"MANIFEST.json")["existing_passed"],
            measured_target_Qpi_at_delivery=self.manifest["new_actual_target_Qpi_labels"])

