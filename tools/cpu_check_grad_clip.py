"""Analyze logged pre-clip update norms, with no GPU or checkpoint loading."""
import json,statistics
from diagnose_state_matched_cpu import OUT
p=OUT/"CPU_CREDIT_DIAGNOSTIC.json"
d=json.loads(p.read_text(encoding="utf-8"))
rows=d["credit_mechanism"]["groups"]
allstats={}
for kind in ("supported","unsupported","all"):
    xs=[r for r in rows if kind=="all" or bool(r["pCCE_supported"])==(kind=="supported")]
    rec={
        "n_groups":len(xs),
        "A_mean_norm":statistics.mean(r["A_grad_norm"] for r in xs),
        "B_mean_norm":statistics.mean(r["B_grad_norm"] for r in xs),
        "A_norm_gt_1":sum(r["A_grad_norm"]>1 for r in xs),
        "B_norm_gt_1":sum(r["B_grad_norm"]>1 for r in xs),
        "B_only_clipped":sum(r["B_grad_norm"]>1>=r["A_grad_norm"] for r in xs),
        "A_only_clipped":sum(r["A_grad_norm"]>1>=r["B_grad_norm"] for r in xs),
        "A_avg_min_1_over_norm":statistics.mean(min(1.,1/r["A_grad_norm"]) for r in xs),
        "B_avg_min_1_over_norm":statistics.mean(min(1.,1/r["B_grad_norm"]) for r in xs),
    }
    allstats[kind]=rec
print("PER_GROUP i support A_norm B_norm A_clip B_clip")
for r in rows:
    print(r["group"], int(r["pCCE_supported"]),
          round(r["A_grad_norm"],4),round(r["B_grad_norm"],4),
          int(r["A_grad_norm"]>1),int(r["B_grad_norm"]>1))
print(json.dumps(allstats,ensure_ascii=False,indent=2))
(OUT/"CPU_GRAD_CLIP_DIAGNOSTIC.json").write_text(
    json.dumps({"clips_by_group":allstats,"per_group":[
        {"i":r["group"],"support":r["pCCE_supported"],
         "A_preclip_norm":r["A_grad_norm"],"B_preclip_norm":r["B_grad_norm"],
         "A_clipped":r["A_grad_norm"]>1,"B_clipped":r["B_grad_norm"]>1,
         "A_scale":min(1,1/r["A_grad_norm"]),
         "B_scale":min(1,1/r["B_grad_norm"])} for r in rows],
         "interpretation":"Norm clipping scales the complete gradient, so an additive loss may still change how much the terminal direction is applied. Logged norms cannot separate neural gradient components or establish causality."},
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
