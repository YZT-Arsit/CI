from pathlib import Path
import tarfile,json
b=Path(r'D:\research\longhistory\artifacts\20261119_composition_signals\agent_rl_v1\policy_qpi_01')
for n in ('state_directed_4b_probe_backup.tar.gz','state_restart_probe16_backup.tar.gz','t4_frozen_probe_backup.tar.gz'):
 print('\nARCHIVE',n)
 with tarfile.open(b/n,'r:gz') as t:
  names=t.getnames()
  print('\n'.join(names[:35]))
  eps=[x for x in names if x.endswith('/episodes.jsonl')]
  for e in eps:
   rows=[json.loads(x) for x in t.extractfile(e).read().decode().splitlines() if x.strip()]
   print('EPISODES',e,len(rows),'policy IDs',len(set(r['policy_id'] for r in rows)))
   print('VERDICT FIRST', [d['verdict'] for d in rows[0]['decisions']])
