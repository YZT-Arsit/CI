"""Verify actual KCI chain executes with one shared action-policy object."""
from pathlib import Path
import sys,inspect
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_shared_semantic_native_rollout import episode
from compare_kci_chain_methods_cpu_action_interaction import GROUP

class SharedRuleSanity:
    """Dummy legal-action policy: tests shared routing, not LLM competence."""
    def decide(self,obs):
        options=obs["legal_actions"]
        for act in ("A_SEND_PACKAGE_A","OPEN_PRIVATE_CHANNEL","VERIFY_PRIVATE_R","COMMIT_RESULT"):
            if act in options:return act,{a:float(a==act) for a in options}
        return options[0],{a:float(a==options[0]) for a in options}
def test_same_policy_object_controls_all_critical_roles():
    world=GROUP["train"][0]
    res=episode({**world,"split":"train"},SharedRuleSanity(),"shared")
    assert res["safe_success"]
    assert not res["CI_violation"]
    decisions=[x for x in res["trace"] if x["policy_source"]=="shared_one_LoRA"]
    assert len(decisions)>=3
    roles={x["role"] for x in decisions}
    assert "RecordsAgent" in roles and "Coordinator" in roles
    assert len({x["role"] for x in decisions})>=3
    assert all(x["native_verdict"]=="VALID" for x in res["trace"])
def test_no_role_specific_switch_in_new_execution_path():
    from kci_shared_semantic_action import SharedSemanticPolicy
    from kci_shared_semantic_native_rollout import episode
    src=inspect.getsource(SharedSemanticPolicy)+inspect.getsource(episode)
    assert "set_adapter" not in src and "load_adapter" not in src
    assert "routed_role" not in src
