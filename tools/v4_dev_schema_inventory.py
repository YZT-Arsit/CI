"""V4 metadata-only inventory. Never deserialize DG36 final_test content."""
import json, pathlib, collections
src=pathlib.Path(r"D:\research\kci_data_v4_pilot_20261122\artifacts\20261122_kci_data_v4_pilot_v1")
names=["TRAIN_SAME_STATE_PREFERENCES.jsonl","TRAIN_V4_HISTORY_EVALUATOR_COMPAT.jsonl",
"MODEL_INPUTS_EVAL_ONLY.jsonl","NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl","CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl",
"EXACT_NATIVE_FUTURE_TRAIN_ONLY.jsonl"]
for name in names:
    path=src/name
    data=[]
    n_skipped=0
    for line in path.open(encoding="utf-8"):
        # Strictly no parsing of final_test DG36
        if '"DG36' in line or '"final_test"' in line:
            n_skipped+=1;continue
        data.append(json.loads(line))
    print("\nFILE",name,"allowed",len(data),"excluded_untouched",n_skipped,flush=True)
    if not data:continue
    print("KEYS",sorted(data[0]),flush=True)
    if name.startswith("TRAIN"):
        for col in ("root","root_id","role","acting_role","stage","category","branch","chosen_action","rejected_action"):
            v=collections.Counter(str(x.get(col)) for x in data)
            if v!={"None":len(data)}:print("GROUP",col,dict(v),flush=True)
        print("SAMPLE",json.dumps({k:v for k,v in data[0].items() if k not in ("prompt","native_evidence","evidence")},ensure_ascii=False)[:6000],flush=True)
    if name=="MODEL_INPUTS_EVAL_ONLY.jsonl":
        print("FIRST_ALLOWED",json.dumps(data[0],ensure_ascii=False)[:6500],flush=True)
        for col in ("root_id","root","split","category","condition","role","stage","variant","history_type"):
            v=collections.Counter(str(x.get(col)) for x in data)
            if v!={"None":len(data)}: print("GROUP",col,dict(v),flush=True)
    if name=="NATIVE_TRAJECTORIES_EVAL_ONLY.jsonl":
        print("FIRST_ALLOWED_META",json.dumps({k:str(v)[:400] for k,v in data[0].items() if k not in ("trace","steps","trajectory","receipts","native_state")},ensure_ascii=False)[:6500],flush=True)
    if name=="CONTROLLED_CONTEXT_PAIRS_EVAL_ONLY.jsonl":
        print("FIRST_ALLOWED_META",json.dumps(data[0],ensure_ascii=False)[:5000],flush=True)
    if name=="EXACT_NATIVE_FUTURE_TRAIN_ONLY.jsonl":
        print("FIRST_ALLOWED_META",json.dumps({k:str(v)[:300] for k,v in data[0].items() if k not in ("trace","steps","trajectory")},ensure_ascii=False)[:7000],flush=True)
