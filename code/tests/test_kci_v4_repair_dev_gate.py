"""Research gate prevents premature dev selection/CCE/DPR work."""
from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"tools/kci_v4_pairwise_repair_dev_eval.py"
def test_dev_evaluation_refuses_unfitted_train_gate(tmp_path):
    gate=tmp_path/"fail.json"
    gate.write_text(json.dumps({"model_method":"grouped_ce",
              "fit_gate_passed":False,"DG34_DG35_DG36_NOT_READ":True}),encoding="utf8")
    # No adapter/base paths need to be valid: must refuse before any model or dev load.
    r=subprocess.run([sys.executable,"-B",str(SCRIPT),
         "--base","DO_NOT_LOAD","--adapter","DO_NOT_LOAD","--method","grouped_ce",
         "--train-gate",str(gate),"--source-data","DO_NOT_OPEN",
         "--output",str(tmp_path/"dev_output")],
         text=True,capture_output=True,timeout=10)
    assert r.returncode!=0 and "Never touch dev" in r.stderr
    assert not (tmp_path/"dev_output").exists()
