"""Bounded token-input diagnostics, after dependency import/setup."""
import os,sys,time,json,threading
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path.insert(0,json.loads((R/'wsl_environment.json').read_text())['private_package_path'])
sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from token_input_v3 import token_encode,augment_pairs
from experiments.receipt_dataset import make_world,SOURCE
from kci.receipt_tasks import ReceiptTask

timer=threading.Timer(9.5,lambda:os._exit(124));timer.daemon=True;timer.start();t=time.perf_counter()
checks=0
for seed in range(32):
    env=ReceiptTask(make_world(140000+seed,depth=1+seed%3,noise=seed%4),SOURCE)
    obs=env.observation(env.initial());a,b=env.actions(env.initial())
    ids=np.array([token_encode(obs,a),token_encode(obs,b)])
    assert ids[0,1]==ids[1,1]
    assert np.count_nonzero(ids[0])==np.count_nonzero(ids[1])
    aug=augment_pairs(ids,np.random.default_rng(seed))
    assert aug[0,1]==aug[1,1]
    mapping={}
    for side in (0,1):
        old=ids[side];new=aug[side]
        # Type counts/flags/length are invariant; relation triples are shuffled.
        assert np.array_equal(old[2:12],new[2:12])
        assert np.count_nonzero(old)==np.count_nonzero(new)
        assert len(set(old[old>=64]))==len(set(new[new>=64]))
        assert sorted(old[(old>=30)&(old<34)])==sorted(new[(new>=30)&(new<34)])
    # Fresh test symbols are entirely disjoint atomic embedding indices.
    fresh=token_encode(obs,a,fresh=True)
    assert set(fresh[fresh>=64]).isdisjoint(ids[0][ids[0]>=64])
    checks+=8
timer.cancel()
report={'worlds':32,'checks':checks,'seconds':time.perf_counter()-t,'all_passed':True,
 'scope':'input equality/type/length invariants; not a learned generalization result'}
(R/'encoding_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
