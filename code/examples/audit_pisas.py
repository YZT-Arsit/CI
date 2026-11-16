"""Use the original four JSON files to check what PiSAs actually supports.

Usage: python examples/audit_pisas.py /path/to/meeting_allocation/scenario_01
No historical reward records are fabricated from scenario.json/timeline.
"""
from pathlib import Path
import json,sys
from kci.datasets import audit_pisas_files

def main(folder):
    root=Path(folder)
    data={name:json.loads((root/name).read_text(encoding='utf-8')) for name in
          ('scenario.json','utility.json','appropriateness.json','visibility.json')}
    audit=audit_pisas_files(data)
    print(json.dumps({'scenario':data['scenario.json'].get('scenario_id'),
      'task':data['scenario.json']['task']['type'],
      'timeline_artifacts':len(data['scenario.json'].get('timeline',[])),
      'static_attribute_labels':len(data['appropriateness.json']['attributes']),
      'is_directly_ready_for_KCI_future_rewards':audit.train_ready,
      'can_reuse':audit.reusables,'must_create':audit.must_create},
      ensure_ascii=False,indent=2))

if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python examples/audit_pisas.py SCENARIO_DIR')
    main(sys.argv[1])
