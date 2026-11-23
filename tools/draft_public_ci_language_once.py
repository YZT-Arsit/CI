r"""One API request for non-authoritative natural-language paraphrase candidates.
Uses ONLY fictional/public task patterns. Never returns or logs the API key.
Output is REVIEW_REQUIRED, never consumed by the executable oracle or training.
Do not retry automatically. Parsed response is saved before validation.
"""
from pathlib import Path
import argparse,json,os,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import PUBLIC_TASKS,SEEDS,graph_texts

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",type=Path,required=True)
    ap.add_argument("--model",default="gpt-5.4-mini")
    ap.add_argument("--max-calls",type=int,default=1)
    args=ap.parse_args()
    if args.max_calls!=1:raise ValueError("Exactly one API call allowed")
    if not os.getenv("OPENAI_API_KEY"):raise RuntimeError("No configured OPENAI_API_KEY")
    out=args.dataset.resolve()
    if not out.is_dir():raise FileNotFoundError(out)
    rawpath=out/"API_TEXT_RAW_CANDIDATES_REVIEW_REQUIRED.json"
    saved=out/"API_LANGUAGE_REVIEW_REQUIRED.jsonl"
    usage=out/"API_LANGUAGE_USAGE.json"
    if any(p.exists() for p in (rawpath,saved,usage)):
        raise FileExistsError("Prior API output exists; never auto overwrite or retry")
    from openai import OpenAI
    from pydantic import BaseModel,Field,ConfigDict
    class Draft(BaseModel):
        model_config=ConfigDict(extra="forbid")
        task_id:str
        user_request_paraphrase:str
        clues:list[str]=Field(min_length=4,max_length=4)
    class Batch(BaseModel):
        model_config=ConfigDict(extra="forbid")
        drafts:list[Draft]=Field(min_length=10,max_length=10)
    requests=[]
    for s in PUBLIC_TASKS:
        clues=graph_texts(s,SEEDS[s.seed_id])
        requests.append({
            "task_id":s.id,"business_workflow":s.title,
            "user_request":s.user_request,
            "source_document_sentences":[clues[f] for f in ("f0","f1","f2","f3")],
            "privacy_category":s.category
        })
    client=OpenAI(timeout=70.0,max_retries=0)
    response=client.responses.parse(
        model=args.model,
        input=[
            {"role":"system","content":
             "Write distinct natural English business-message paraphrases for ten entirely fictional test cases. "
             "Keep the same factual entailment exactly: preserve all subject names, case identifiers, private-reference identifiers, "
             "the direction and type of each link, which message mentions the restricted category, and the independent content of each clue. "
             "Do not combine separate clues or disclose the person's restricted category in any single sentence. "
             "Do not add rules, policy permissions, tasks, correct actions, hints, reward labels, extra events, or claims. "
             "For each case, rewrite the user request and four source sentences in their original order. "
             "Varied natural phrasing is welcome only where semantics stays invariant. Output only the structured schema."},
            {"role":"user","content":json.dumps(requests,ensure_ascii=False)}
        ],
        text_format=Batch,
        max_output_tokens=6000,
        store=False
    )
    parsed=response.output_parsed
    u=getattr(response,"usage",None)
    stat={"model":args.model,"api_calls":1,"store":False,
          "input_tokens":getattr(u,"input_tokens",None),
          "output_tokens":getattr(u,"output_tokens",None),
          "total_tokens":getattr(u,"total_tokens",None),
          "response_id":getattr(response,"id",None),
          "no_user_private_data_sent":True,
          "model_training_or_label_oracle_modified":False}
    # Save raw parsed output FIRST. Never lose paid content to later checks.
    rawpath.write_text(json.dumps({
        "api_usage":stat,
        "raw_parsed_drafts":parsed.model_dump() if parsed is not None else None,
        "review_required":True
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if parsed is None:
        stat["status"]="INCOMPLETE_RAW_RESPONSE_SAVED"
        usage.write_text(json.dumps(stat,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(stat,ensure_ascii=False),flush=True)
        return
    submitted={x.id:x for x in PUBLIC_TASKS}
    drafts=parsed.drafts
    unique={x.task_id for x in drafts}
    valid=len(drafts)==10 and unique==set(submitted)
    rows=[]
    for x in drafts:
        s=submitted.get(x.task_id)
        if s is None:continue
        org=graph_texts(s,SEEDS[s.seed_id])
        original=[org[f] for f in ("f0","f1","f2","f3")]
        words=[x.user_request_paraphrase]+x.clues
        has_ids_all=all(
            set(__import__("re").findall(r"[WR]\d\d-[A-Z]\d+",src)).issubset(
                set(__import__("re").findall(r"[WR]\d\d-[A-Z]\d+",dst)))
            for src,dst in zip(original,x.clues))
        total_ok=all(isinstance(v,str) and len(v)>25 for v in words)
        field_ok=(has_ids_all and total_ok)
        rows.append({
            "task_id":x.task_id,"upstream_source":s.reference,
            "user_request_original":s.user_request,
            "user_request_candidate":x.user_request_paraphrase,
            "four_original_source_messages":original,
            "four_candidate_source_messages":x.clues,
            "preliminary_identifier_preservation":has_ids_all,
            "preliminary_length_validation":total_ok,
            "candidate_structurally_usable":field_ok,
            "manual_semantic_entailment_review":"PENDING",
            "manually_approved":False,
            "model_inputs_changed":False,
            "evaluator_rules_changed":False,
            "training_eligible":False
        })
        valid=valid and field_ok
    with saved.open("x",encoding="utf-8",newline="\n") as h:
        for rec in rows:h.write(json.dumps(rec,ensure_ascii=False,sort_keys=True)+"\n")
    stat.update({"status":"REVIEW_CANDIDATES_SAVED" if valid else "CANDIDATES_REQUIRE_REPAIR",
                 "n_candidates":len(rows),
                 "all_preliminary_structure_checks_pass":bool(valid),
                 "source_truth_or_official_data_modified":False})
    usage.write_text(json.dumps(stat,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(stat,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__":main()
