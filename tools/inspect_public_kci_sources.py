import ast, json, re
from pathlib import Path
a=Path(r"D:\research\kci_public_refs_20261119\agentdojo\src\agentdojo\default_suites\v1\workspace\user_tasks.py").read_text(encoding="utf-8")
tree=ast.parse(a)
for node in tree.body:
 if isinstance(node,ast.ClassDef) and re.match(r"UserTask\d+",node.name):
  p=[x for x in node.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="PROMPT" for t in x.targets)]
  if not p:continue
  val=p[0].value
  try:prompt=ast.literal_eval(val)
  except:prompt=ast.get_source_segment(a,val)
  tools=re.findall(r'function="([^"]+)"',ast.get_source_segment(a,node) or "")
  print(node.name,repr(str(prompt)[:200]),"TOOLS",tools[:8])
print("\nTAU RETAIL")
data=json.loads(Path(r"D:\research\kci_public_refs_20261119\tau2-bench\data\tau2\domains\retail\tasks.json").read_text(encoding="utf-8"))
print(type(data).__name__,list(data.keys())[:8] if isinstance(data,dict) else len(data))
if isinstance(data,dict):data=list(data.values())[0]
for d in data[:15]:
 print("id",d.get("id"),"keys",list(d.keys()),"user",str(d.get("user_scenario",{}))[:380],"check",str(d.get("evaluation_criteria",{}))[:260])
