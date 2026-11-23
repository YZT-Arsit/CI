"""Regression gates for the sole recipient-history evaluator ablation.
Checks natural-language rules, actual first-disclosure state and ablation consistency.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from compare_kci_chain_methods_cpu_action_interaction import GROUP
from kci_exact_policy_continuation_cpu import environment_public
from kci_llm_text_surface import render_role_text
from kci_history_eval_dev_probe_20261122 import at_first_release,judge
RAW="SEND_RAW_R"
VERIFY="VERIFY_PRIVATE_R"

def test_inference_renders_intermediate_conclusions_as_intermediate():
    affected=Counter()
    roots=set()
    for split,worlds in GROUP.items():
        for world in worlds:
            env=environment_public(world)
            s=at_first_release(env,next(iter(env.records)))
            observation=env.observation(s)
            rules=observation["declared_knowledge_rules"]
            prompt=render_role_text(observation)
            assert prompt.count("R can infer the restricted association.")==sum(r["entails"]=="protected association" for r in rules)
            for rule in rules:
                if rule["entails"] in ("X","Y","Z"):
                    assert "R can infer intermediate fact "+rule["entails"]+" (not itself a restricted association)." in prompt
                    affected[world["root"]]+=1
            roots.add(world["root"])
    assert len(roots)==7 and len(affected)==5

def test_two_labelers_do_not_expose_hidden_history_to_actor():
    samples=[]
    for w in (GROUP["train"][0],GROUP["exploratory_test"][0]):
        env=environment_public(w)
        for kind,facts in (("risky",env.spec.risky_prior),("safe",env.spec.safe_prior)):
            name=next(k for k,v in env.records.items() if tuple(v["facts"])==tuple(facts))
            state=at_first_release(env,name)
            obs=env.observation(state)
            prompt_before=render_role_text(obs)
            seen,_,traces=judge(env,state,True)
            blind,_,blind_traces=judge(env,state,False)
            assert seen in (RAW,VERIFY) and blind in (RAW,VERIFY)
            assert render_role_text(env.observation(state))==prompt_before
            assert obs["legal_actions"]==env.observation(state)["legal_actions"]
            assert not state.native.done
            samples.append((w["root"],kind,seen,blind,traces,blind_traces))
    assert len(samples)==4
    assert all(x[3]==RAW for x in samples)

def test_frozen_paired_teacher_arms_identical_actor_inputs():
    base=ROOT/"artifacts"/"20261122_history_evaluator_dev_v1"
    if not base.exists():
        return  # Core repo test can run on published checkout lacking private data
    a=[json.loads(x) for x in (base/"TRAIN_VISIBLE.jsonl").read_text(encoding="utf8").splitlines()]
    b=[json.loads(x) for x in (base/"TRAIN_BLIND.jsonl").read_text(encoding="utf8").splitlines()]
    assert len(a)==len(b)==32
    assert len({x["root"] for x in a})==4
    for x,y in zip(a,b):
        assert (x["prompt"],x["observation_hash"],x["state_hash"],x["all_executable_actions"])==(
            y["prompt"],y["observation_hash"],y["state_hash"],y["all_executable_actions"])
        assert x["actual_reference_label"]==y["actual_reference_label"]
        if x["branch"]=="risky" and x["root"] in ("DG01","DG05"):
            assert x["chosen_action"]==VERIFY and y["chosen_action"]==RAW
