"""Freeze native V4 full-agent DEV worlds exclusively; DG36 never parsed."""
from pathlib import Path
from hashlib import sha256
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"tools")]
from kci_v4_loader import checked_read_dev
SRC=Path(r"D:\research\kci_data_v4_pilot_20261122\artifacts\20261122_kci_data_v4_pilot_v1")
OUT=ROOT/"artifacts/20261123_v4_semantic_control_v1"
def main():
    roots=checked_read_dev(SRC/"ROOTS.jsonl")
    worlds=checked_read_dev(SRC/"WORLD_CONFIGS.jsonl")
    assert len(roots)==2 and len(worlds)==10
    assert {r["root_id"] for r in roots}=={"DG34","DG35"}
    assert {r["root_id"] for r in worlds}=={"DG34","DG35"}
    rootid={r["root_id"] for r in roots}
    assert all(w["root_id"] in rootid for w in worlds)
    hashes={}
    for f,rows in [("DEV_ROOT_SPECS.jsonl",roots),("DEV_WORLD_CONFIGS.jsonl",worlds)]:
        path=OUT/f
        assert not path.exists()
        path.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows),encoding="utf8")
        hashes[f]=sha256(path.read_bytes()).hexdigest()
    print(json.dumps({"DEV_V4_FULL_NATIVE_FREEZE":True,"roots":list(sorted(rootid)),"worlds":len(worlds),
                      "DG36_not_parsed":True,"hashes":hashes},ensure_ascii=False))
if __name__=="__main__":main()
