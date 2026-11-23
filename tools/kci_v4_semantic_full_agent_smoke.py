"""Child-process-only native V4 full trajectory regression smoke."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"code"),str(ROOT/"tools")]
from kci_v4_semantic_full_agent import run_env,spec_table,read
DATA=ROOT/"artifacts/20261123_v4_semantic_control_v1"
class AllVerify:
    def distribution(self,prompt,actions,method):
        assert "Stage: release" in prompt
        assert "VERIFY_PRIVATE_R" in actions
        return {a:(0.98 if a=="VERIFY_PRIVATE_R" else 0.01) for a in actions},{a:0 for a in actions}
def main():
    specs=spec_table(DATA)
    worlds=read(DATA/"DEV_WORLD_CONFIGS.jsonl")
    assert len(worlds)==10
    for w in worlds:
        native=run_env(specs,w,"aware_ce","native","controlled_release",AllVerify())
        alias=run_env(specs,w,"aware_ce","entity_alias","controlled_release",AllVerify())
        assert native["native_actual_success"] and alias["native_actual_success"]
        assert not native["native_actual_violation"]
        assert native["native_execution_steps"]==alias["native_execution_steps"]
        assert native["model_decision_count"]>=1
    print("V4_DEV_FULL_NATIVE_10_WORLDS_TEST_PASSED")
if __name__=="__main__":main()
