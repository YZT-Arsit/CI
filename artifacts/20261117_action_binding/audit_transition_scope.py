"""Count actual prior execution transitions separately from relational path depth."""
import json,hashlib
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parent;OLD=R.parent/'20261116_relation_solution'

def run():
    rows=[];hashes={}
    for seed in (17,29,41):
        file=OLD/'representation_repair'/f'local_{seed}_width64.jsonl';hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        for row in map(json.loads,file.read_text(encoding='utf-8').splitlines()):
            if row['depth']!=32:continue
            rows.append({'seed':seed,'world_id':row['world_id'],'relation_depth':row['depth'],'policy_decisions':len(row['trace']),'environment_transitions':row['final']['diagnostic_steps'],'decision_roles':[t['role'] for t in row['trace']],'ledger_messages_including_initial_history':len(row['final']['ledger']),'resource_consumptions_inside_actions':len(row['final']['consumption_log'])})
    assert rows and all(r['policy_decisions']==r['environment_transitions']==3 for r in rows)
    result={'records':rows,'source_sha256':hashes,'episodes':len(rows),'relation_depth':32,'decision_count_distribution':dict(Counter(r['policy_decisions'] for r in rows)),'proved_32_step_collaboration':False,'interpretation':'32 static relation links were reasoned over during a three-decision Planner/Source/Worker protocol; bundled messages/reads are not additional model decisions'}
    (R/'TRANSITION_SCOPE.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256')}))

if __name__=='__main__':run()
