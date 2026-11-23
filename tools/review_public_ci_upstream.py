"""Re-open pinned upstream sources and inspect whether adapted business tasks are faithful.
No external API calls and no modifications to the original public benchmark.
"""
from pathlib import Path
import ast,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code"))
from kci.public_workflow_ci import PUBLIC_TASKS,AGENTDOJO_SHA,TAU_SHA
UP=Path(r"D:\research\kci_public_refs_20261119")
work=UP/"agentdojo"/"src"/"agentdojo"/"default_suites"/"v1"/"workspace"/"user_tasks.py"
retail=UP/"tau2-bench"/"data"/"tau2"/"domains"/"retail"/"tasks.json"
assert work.is_file() and retail.is_file()
assert subprocess.check_output(["git","-C",str(UP/"agentdojo"),"rev-parse","HEAD"],text=True).strip()==AGENTDOJO_SHA
assert subprocess.check_output(["git","-C",str(UP/"tau2-bench"),"rev-parse","HEAD"],text=True).strip()==TAU_SHA
src=work.read_text(encoding="utf-8")
classes={x.name:x for x in ast.parse(src).body if isinstance(x,ast.ClassDef)}
tasks={str(x["id"]):x for x in json.loads(retail.read_text(encoding="utf-8"))}
data=[]
for sp in PUBLIC_TASKS:
    if sp.suite=="agentdojo_workspace":
        cls=classes["UserTask"+sp.upstream_id]
        code=ast.get_source_segment(src,cls)
        prompt=[a.value for a in cls.body if isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PROMPT" for t in a.targets)]
        assert prompt
        source_prompt=ast.get_source_segment(src,prompt[0])[:620]
        calls=re.findall(r'function="([^"]+)"',code)
        evidence={
            "pilot_id":sp.id,"suite":sp.suite,"source_class":"UserTask"+sp.upstream_id,
            "source_request_template":source_prompt,
            "source_tool_calls":calls,
            "source_has_utility_function":"def utility(" in code,
            "source_changes_environment":"post_environment" in code,
            "pilot_goal":sp.user_request,"pilot_reads":list(sp.read_tools),
            "pilot_writes":list(sp.commit_tools)}
    else:
        t=tasks[sp.upstream_id]
        ins=t.get("user_scenario",{}).get("instructions",{})
        acts=t.get("evaluation_criteria",{}).get("actions",[])
        evidence={
            "pilot_id":sp.id,"suite":sp.suite,"source_task_id":sp.upstream_id,
            "source_user_request":ins.get("reason_for_call"),
            "source_tool_calls":[x["name"] for x in acts],
            "source_confirmation_instructions":str(ins.get("task_instructions",""))[:360],
            "source_has_DB_evaluation":bool(t.get("evaluation_criteria")),
            "pilot_goal":sp.user_request,"pilot_reads":list(sp.read_tools),
            "pilot_writes":list(sp.commit_tools)}
    data.append(evidence)
out=ROOT/"artifacts"/"20261119_composition_signals"/"public_workflow_ci_10_final"/"FIRST_PASS_UPSTREAM_EXCERPTS.json"
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps({"pinned_sources":{"agentdojo":AGENTDOJO_SHA,"tau2":TAU_SHA},
                           "cases":data},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for x in data:
    print(x["pilot_id"],"source=",x.get("source_class",x.get("source_task_id")),
          "source_tools=",",".join(sorted(set(x["source_tool_calls"]))),
          "pilot_goal=",x["pilot_goal"],flush=True)
