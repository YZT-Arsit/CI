"""V4 native module uses process-local graph registration; isolate it."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
def test_actual_native_team_executes_in_isolated_process():
    smoke=ROOT/"tools/kci_v4_semantic_full_agent_smoke.py"
    result=subprocess.run([sys.executable,"-B",str(smoke)],
        capture_output=True,text=True,timeout=35,check=True)
    assert "V4_DEV_FULL_NATIVE_10_WORLDS_TEST_PASSED" in result.stdout
