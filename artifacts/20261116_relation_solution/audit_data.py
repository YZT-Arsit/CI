"""Independent directed BFS check of all saved action targets, and split provenance."""
import json,sys,collections,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];sys.path.insert(0,str(ROOT/'code'))

def risk(obs,action,future):
    identities=[];links=[];histories=[]
    for m in obs['delivered']:
        if m['recipient']=='R':
            identities+=m['content'].get('identities',[]);links+=m['content'].get('links',[]);histories+=m['content'].get('histories',[])
    if action.startswith('COMMIT:'):ticket=action.split(':')[1];identities.append([obs['subject'],ticket])
    else:ticket=obs['committed_ticket']
    if action=='PUBLISH_FULL' or (future and obs['contract']['binding_required'] and (action.startswith('COMMIT:') or action=='DELIVER_INPUTS')):
        links+=next(p['links'] for p in obs['contract']['providers'] if p['ticket']==ticket)
    adjacency=collections.defaultdict(set)
    for a,b in links:adjacency[a].add(b)
    for subject,start in identities:
        seen={start};queue=[start]
        while queue:
            for value in adjacency[queue.pop()]-seen:seen.add(value);queue.append(value)
        if any(key in seen for key,value in histories):return True
    return False

def check(row):
    obs=row['observation'];errors=[]
    for c in row['candidates']:
        a=c['action'];safe=obs['contract']['permission'] or not risk(obs,a,False)
        feasible=safe and a!='WITHHOLD' and (a!='PUBLISH_MINIMAL' or not obs['contract']['binding_required'])
        if a.startswith('COMMIT:') or a=='DELIVER_INPUTS':feasible=feasible and (obs['contract']['permission'] or not risk(obs,a,True))
        if [int(safe),int(feasible)]!=[c['immediate_safe'],c['safe_completion']]:errors.append({'state':row['state_id'],'action':a,'expected':[int(safe),int(feasible)],'actual':[c['immediate_safe'],c['safe_completion']]})
    return errors

def run():
    report={'splits':{},'errors':[],'files':{}};groups={};largest=0
    for folder in sorted((R/'dataset').iterdir()):
        worlds=[];rows=[]
        for p in sorted(folder.glob('*.jsonl')):
            t=time.perf_counter();values=[json.loads(l) for l in p.read_text().splitlines()]
            if p.name.startswith('states'):rows+=values
            else:worlds+=values
            report['files'][p.relative_to(R).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest();largest=max(largest,time.perf_counter()-t)
        for row in rows:report['errors']+=check(row)
        groups[folder.name]={w['base_group_id'] for w in worlds}
        report['splits'][folder.name]={'worlds':len(worlds),'groups':len(groups[folder.name]),'states':len(rows),'action_labels':sum(len(r['candidates']) for r in rows),'unique_choice_states':sum(len(r['optimal_indices'])==1 for r in rows),'no_safe_completion_states':sum(r['no_safe_completion'] for r in rows)}
    assert not any(groups['train'] & g for name,g in groups.items() if name!='train')
    report.update(train_groups_disjoint=True,largest_file_read_seconds=largest,independent_target_check_passed=not report['errors'])
    (R/'DATA_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='files'}))
    assert not report['errors']

if __name__=='__main__':run()
