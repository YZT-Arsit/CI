"""Describe invalid outputs without relaxing the frozen parser or changing scores."""
import collections,json,re
from pathlib import Path
R=Path(__file__).resolve().parent/'risk_repair'

def main():
    result={};reports=json.loads((R/'RESULTS.json').read_text())['summaries']
    for arm in ['utility','safety_utility']:
        total=0;invalid=[]
        for report in reports:
            if report['arm']!=arm:continue
            for line in (R/arm/(report['condition']+'.jsonl')).read_text().splitlines():
                row=json.loads(line)
                for entry in row['trace']:
                    total+=1
                    if not entry['format_valid']:invalid.append({'condition':report['condition'],'world_id':row['world_id'],'role':entry['role'],'raw':entry['raw'],'seconds_batch':entry['seconds_batch'],'over_time_bound':entry['seconds_batch']>=9.5,'bare_legal_letter_only':bool(re.fullmatch('[A-C]',entry['raw'].strip()))})
        result[arm]={'action_outputs':total,'invalid_outputs':len(invalid),'over_time_bound':sum(r['over_time_bound'] for r in invalid),'bare_letter_outputs_still_counted_invalid':sum(r['bare_legal_letter_only'] for r in invalid),'invalid_raw_counts':dict(collections.Counter(r['raw'] for r in invalid)),'records':invalid}
    (R/'FORMAT_AUDIT.json').write_text(json.dumps({'scope':'post-run error attribution only; frozen strict scores unchanged','arms':result},indent=2));print(json.dumps({arm:{k:v for k,v in row.items() if k not in ['records','invalid_raw_counts']} for arm,row in result.items()}))

if __name__=='__main__':main()
