"""V4 loader schema/paired sample tests. DG36 never parsed."""
from pathlib import Path
import sys,json,hashlib,collections
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_v4_loader import (V4_DEFAULT,strict_train_groups,native_dev_rows,
                           prepared_manifest,option_lines,RELEASE_CATEGORIES)
from kci_v4_train_4b import read_groups,model_input,objective_loss,lmargin
TRAIN=ROOT/"artifacts/20261123_v4_release_control_v1/TRAIN_RELEASE_GROUPS.jsonl"
def test_all_v4_train_groups_role_stage_order():
    g=strict_train_groups()
    assert len(g)==18
    assert sum(x["stage"]=="release" for x in g)==9
    assert set(x["role"] for x in g)=={"ComplianceAgent","RecordsAgent","Coordinator"}
    assert set(x["stage"] for x in g)=={"release","preflight","A_SEND"}
    for x in g:
        assert len(x["paired_option_records"])==2
        for rec in x["paired_option_records"]:
            assert rec["option_order"].index(rec["chosen_action"])==ord(rec["chosen_letter"])-65
            assert rec["option_order"].index(rec["rejected_action"])==ord(rec["rejected_letter"])-65
def test_release_groups_distinct_and_no_double_swaps():
    r=read_groups(TRAIN,True)
    assert len(r)==9
    assert sum(x["n_original_records"] for x in r)==18
    assert collections.Counter(x["root"] for x in r)=={"DG31":3,"DG32":3,"DG33":3}
    assert all(x["category"] in RELEASE_CATEGORIES for x in r)
    for x in r:
        a,b=x["paired_option_records"]
        assert a["original_option_body"]==b["original_option_body"]
        assert a["option_order"]!=b["option_order"]
        assert model_input(a)!=model_input(b)
def test_dev_only_DG34_DG35_and_fixed_native():
    dev,pairs=native_dev_rows()
    assert len(dev)==10 and len(pairs)==8
    assert {x["root"] for x in dev}=={"DG34","DG35"}
    assert {x["root"] for x in pairs}=={"DG34","DG35"}
    assert collections.Counter(x["gold_action"] for x in dev)=={"SEND_RAW_R":8,"VERIFY_PRIVATE_R":2}
    assert all(x["source_split"]=="dev" for x in dev)
    assert all(x["raw_fixed_suffix"]["outcome_grade"]=="verified_native_fixed_suffix_not_Qpi_not_optimal" for x in dev)
    assert all("DG36" not in str(x) for x in dev)
def test_dpo_CE_objective_gradient_with_two_supplied_orders():
    groups=read_groups(TRAIN,True)
    group=groups[0]
    class Mock:
        def __init__(self):
            self.x=torch.nn.Parameter(torch.tensor([0.1,0.2,0.3],dtype=torch.float32))
        def log_distribution(self,prompt,actions):
            # Depend on semantic actions and V4 menu but differentiable.
            values=[self.x[{"SEND_RAW_R":0,"VERIFY_PRIVATE_R":1,"ABORT":2}[a]] for a in actions]
            return torch.log_softmax(torch.stack(values),dim=0)
    p=Mock()
    for obj in ("action_ce","dpo"):
        p.x.grad=None
        loss,margin=objective_loss(p,group,obj,2.0,(torch.tensor(0.),torch.tensor(0.)) if obj=="dpo" else None)
        assert torch.isfinite(loss)
        loss.backward()
        assert p.x.grad.abs().sum()>0
def test_prompt_integrity_and_DG36_guard():
    r=read_groups(TRAIN,True)
    for g in r:
        assert all("DG36" not in x["prompt"] for x in g["paired_option_records"])
        for x in g["paired_option_records"]:
            assert len(hashlib.sha256(x["prompt"].encode()).hexdigest())==64
            assert option_lines(x["prompt"],x["option_order"])==x["original_option_body"]
