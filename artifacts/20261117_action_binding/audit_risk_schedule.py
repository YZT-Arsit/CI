"""Inspect whether the added safety target supplies new action-set information."""
import ast,json
from pathlib import Path
R=Path(__file__).resolve().parent;OLD=R.parent/'20261116_relation_solution'

def main():
    rows=[json.loads(line) for p in sorted((OLD/'dataset/train').glob('states_*.jsonl')) for line in p.read_text().splitlines()]
    node=next(n for n in ast.parse((R/'coverage_repair.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='full_buckets');namespace={'load_states':lambda split:rows};exec(compile(ast.Module(body=[node],type_ignores=[]),'coverage_full_buckets','exec'),namespace);buckets=namespace['full_buckets']();plan=json.loads((R/'risk_repair/PLAN.json').read_text());counts={'states':0,'nonconstant_safety_states':0,'safe_set_equals_optimal_set_when_nonconstant':0,'optimal_strict_subset_of_safe_when_nonconstant':0,'by_role':{}}
    for batch in plan['batches']:
        for row in buckets[tuple(batch['key'])][batch['index']]:
            safe={i for i,c in enumerate(row['candidates']) if c['immediate_safe']};good=set(row['optimal_indices']);assert good<=safe and safe;counts['states']+=1;nonconstant=len(safe)<len(row['candidates']);counts['nonconstant_safety_states']+=nonconstant
            if nonconstant:
                counts['safe_set_equals_optimal_set_when_nonconstant']+=safe==good;counts['optimal_strict_subset_of_safe_when_nonconstant']+=good<safe;role=row['observation']['role'];counts['by_role'][role]=counts['by_role'].get(role,0)+1
    counts['interpretation']='When safe==optimal, the extra term equals the existing optimal-action decision loss minus the negative log legal-letter mass. It reweights an already supervised choice, not a new relational target or new information source.'
    (R/'RISK_SCHEDULE_AUDIT.json').write_text(json.dumps(counts,indent=2));print(json.dumps(counts))

if __name__=='__main__':main()
