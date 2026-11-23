r"""One-call, non-authoritative natural-language draft for KCI seed tasks.

Only public-style fictional task contracts (NOT receiver histories, rewards,
private facts, split labels or correct actions) go to the API. This does not
change or validate the trusted simulator, oracle, policy or dataset splits.
A maximum of exactly ONE request, no automatic retry; store=False.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--model",default="gpt-5.4-mini")
    p.add_argument("--limit-calls",type=int,default=1)
    args=p.parse_args()
    if args.limit_calls!=1:raise ValueError("This pilot must use exactly one bounded request")
    root=args.dataset.resolve()
    if not root.is_dir():raise FileNotFoundError(root)
    out=root/"API_LANGUAGE_CANDIDATE_REVIEW.jsonl"
    usage_file=root/"API_USAGE_NO_SECRETS.json"
    if out.exists() or usage_file.exists():raise FileExistsError("Refuse overwriting a previous API result")
    if not os.getenv("OPENAI_API_KEY"):raise RuntimeError("OPENAI_API_KEY is not set (key contents never displayed)")
    from openai import OpenAI
    from pydantic import BaseModel, Field, ConfigDict
    class Surface(BaseModel):
        model_config=ConfigDict(extra="forbid")
        seed_id: str=Field(min_length=4,max_length=12)
        task_paraphrase: str=Field(min_length=40,max_length=320)
        brief_context: str=Field(min_length=25,max_length=180)
    class Batch(BaseModel):
        model_config=ConfigDict(extra="forbid")
        rows: list[Surface]=Field(min_length=20,max_length=20)
    inputs=[json.loads(x) for x in (root/"roots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(inputs)==20
    payload=[{"id":x["seed_id"],"domain":x["domain"],
              "task_goal":x["task_goal"],
              "required_checks":x["check_meanings"]}
             for x in inputs]
    client=OpenAI(timeout=50.0,max_retries=0)
    resp=client.responses.parse(
        model=args.model,
        input=[
            {"role":"system","content":
             "You are a controlled text editor for a synthetic multi-agent task benchmark. "
             "Produce exactly one concise natural-language paraphrase and one short fictional context for each provided task. "
             "Preserve the verification goal and required checks verbatim in meaning, but keep the prose readable. "
             "DO NOT add new requirements, prices, deadlines, people, numbers, tool capabilities, permissions, privacy norms, "
             "correct actions, rewards, inference results, hints or labels. No real people or companies. "
             "The original structured task contract always remains the authority. "
             "Output only the requested structured fields."},
            {"role":"user","content":json.dumps(payload,ensure_ascii=False)}
        ],
        text_format=Batch,
        max_output_tokens=3300,
        store=False,
    )
    data=resp.output_parsed
    if data is None:raise RuntimeError("API did not return a complete structured response")
    proposals=data.rows
    observed=[x.seed_id for x in proposals]
    expected=[x["id"] for x in payload]
    if len(set(observed))!=20 or set(observed)!=set(expected):raise ValueError("API proposal did not exactly match 20 source seed IDs")
    disallowed=("route_public","route_private","correct action","always choose","gold label","qstar","qpi",
                "prompt injection","ignore previous","secret result","private outcome","chosen action")
    records=[]
    for x in proposals:
        merged=(x.task_paraphrase+" "+x.brief_context).lower()
        if any(z in merged for z in disallowed):raise ValueError("Proposed prose may contain action guidance")
        records.append({"seed_id":x.seed_id,"text_revision_status":"HUMAN_REVIEW_REQUIRED",
                        "task_paraphrase_candidate":x.task_paraphrase,
                        "brief_context_candidate":x.brief_context,
                        "changes_environment_or_labels":False,
                        "may_be_used_for_training":False,
                        "source_contract_sha256":hashlib.sha256(
                             next(v for v in inputs if v["seed_id"]==x.seed_id)["task_goal"].encode()).hexdigest()})
    out.write_text("".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in sorted(records,key=lambda x:x["seed_id"])),encoding="utf-8")
    u=getattr(resp,"usage",None)
    report={"model":args.model,"completed_call_count":1,"store":False,
            "candidate_texts":len(records),"human_equivalence_review_required":True,
            "scoring_oracle_touched":False,
            "input_tokens":getattr(u,"input_tokens",None),
            "output_tokens":getattr(u,"output_tokens",None),
            "total_tokens":getattr(u,"total_tokens",None)}
    usage_file.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ONE_API_CALL_COMPLETED","texts":20,
                      "review_required":True,
                      "usage_tokens":report["total_tokens"],
                      "candidate_file":str(out)},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
