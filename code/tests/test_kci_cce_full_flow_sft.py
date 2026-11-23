"""Action SFT same-root witnessed native success, ABORT remains legal."""
from pathlib import Path
import sys
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"tools"),str(ROOT/"code")]
from kci_cce_full_flow_sft_train import (load,balanced_schedule,STAGE_CYCLE,RUN_STEPS,BATCH)
DATA=ROOT/"artifacts/20261124_cce_downstream_sft_v1/SHARED_ROLE_ACTION_SFT_TRAIN_ONLY.jsonl"
def test_full_success_witnesses_role_state_and_no_oracle():
    rows=load(DATA)
    assert len(rows)==1358
    assert {r["root"] for r in rows}=={"P201","P202","P203","P205","P206","P207","P208"}
    assert set(STAGE_CYCLE)==set(r["stage"] for r in rows)
    assert not any(r["stage"]=="A_SEND" for r in rows)
    assert all(r["chosen_action"]!="ABORT" and "ABORT" in r["all_executable_actions"] for r in rows)
def test_predeclared_root_and_stage_balanced_schedule():
    rows=load(DATA)
    s=balanced_schedule(rows)
    assert len(s)==RUN_STEPS*BATCH==320
    assert all("ABORT" in r["all_executable_actions"] for r in s)
    stages=Counter(r["stage"] for r in s)
    assert all(stages[x]>=20 for x in set(STAGE_CYCLE)),stages
    assert len(set(r["root"] for r in s))==7
    assert [r["same_observation_sha256"] for r in s]==[
        r["same_observation_sha256"] for r in balanced_schedule(rows)]
