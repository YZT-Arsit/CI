"""Reuse native replay audit, then score success across both presentation views."""
import hashlib,json,sys
from pathlib import Path
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R.parent/'20261117_action_binding'))
import audit_learned_evidence as replay

def main():
    plan=json.loads((R/'repair/PLAN.json').read_text());assert all(hashlib.sha256((R/p).read_bytes()).hexdigest()==h for p,h in plan['dependency_hashes'].items());assert hashlib.sha256((R/'repair/dev_worlds.json').read_bytes()).hexdigest()==plan['dev_sha256'];replay.R=R;replay.run('repair');results=[]
    def rows(arm,condition):return [json.loads(line) for line in (R/'repair'/arm/(condition+'.jsonl')).read_text().splitlines()]
    for arm in plan['arms']:
        for family in ['base','depth3','internal3','cycle4']:
            sets=[rows(arm,family+'_v'+str(view)+suffix) for view in (0,1) for suffix in ['', '_swap']];assert len({len(x) for x in sets})==1;robust=0;planner_same=0;planner_valid=0
            for a,b,c,d in zip(*sets):
                assert a['world_id']==c['world_id'] and b['world_id']==d['world_id'];robust+=all(r['strict_success'] for r in [a,b,c,d])
                for x,y in [(a,c),(b,d)]:
                    one,two=x['trace'][0],y['trace'][0];valid=one['format_valid'] and two['format_valid'];planner_valid+=valid;planner_same+=valid and one['selected']==two['selected']
            results.append({'arm':arm,'family':family,'semantic_world_pairs':len(sets[0]),'all_four_episodes_strict_success':robust,'semantic_states':2*len(sets[0]),'planner_both_views_valid':planner_valid,'planner_same_valid_semantic_action_across_views':planner_same})
        for family in ['optional','authorized','draft','both']:
            a=rows(arm,family+'_v0');b=rows(arm,family+'_v1');assert [x['world_id'] for x in a]==[x['world_id'] for x in b];results.append({'arm':arm,'family':family,'semantic_worlds':len(a),'both_views_strict_success':sum(x['strict_success'] and y['strict_success'] for x,y in zip(a,b)),'both_views_valid_ci':sum(x['ci'] and not x['invalid'] and y['ci'] and not y['invalid'] for x,y in zip(a,b))})
    result={'results':results,'scope':'fresh random-world DEV,36basegroups/56semanticworlds/112presentationversions perarm; not source holdout or three seeds','whole_goal_complete':False};(R/'repair/VIEW_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
