"""Re-evaluate saved finite-policy weights against frozen final core; no retraining."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from experiments.preference_probe import features,metrics,trajectory_metrics
from experiments.common import write_result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);args=p.parse_args();t=time.perf_counter()
    rows=[json.loads(x) for x in Path(args.dataset).read_text(encoding='utf-8').splitlines()]
    checks=[]
    for path in sorted(Path(args.folder).glob('*.json')):
        d=json.loads(path.read_text());rep=d['representation'];theta=np.array(d['parameters']);ref=np.array(d['frozen_reference_parameters'])
        x=np.stack([features(r['prompt'],r['chosen'],rep)-features(r['prompt'],r['rejected'],rep) for r in rows])
        actual=metrics(theta,ref,x,rows);traj=trajectory_metrics(theta,rep)
        okay=actual==d['after']
        for old,new in zip(d['trajectory_diagnostic']['rows'],traj['rows']):
            okay=okay and all(old[k]==new[k] for k in ('permitted','horizon','initial','safe_success','ci_compliant','cost'))
            okay=okay and list(old['actions'])==list(new['actions'])
        checks.append({'file':path.name,'sha256':sha256(path.read_bytes()).hexdigest(),'replay_pass':bool(okay),
                       'safe_successes':traj['safe_successes'],'feasible':traj['feasible_episodes'],
                       'violations':traj['confirmed_violations']})
    core=Path(__file__).resolve().parents[1]/'kci'
    result={'seconds':time.perf_counter()-t,'checks':checks,'all_pass':all(c['replay_pass'] for c in checks),
            'core_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(core.glob('*.py'))},
            'dataset_sha256':sha256(Path(args.dataset).read_bytes()).hexdigest(),
            'scope':'saved_weight_replay_only_no_new_fit_or_new_llm_calls'}
    write_result(args.output,result);print({'policies':len(checks),'all_pass':result['all_pass'],'seconds':result['seconds']})
    raise SystemExit(0 if result['all_pass'] else 1)
