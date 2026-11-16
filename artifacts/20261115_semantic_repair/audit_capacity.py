"""Explicit representation limits; no silent long-history truncation."""
import os,sys,time,json,threading
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1]
sys.path.insert(0,json.loads((R/'wsl_environment.json').read_text())['private_package_path']);sys.path.insert(0,str(ROOT/'code'))
from experiments.receipt_dataset import make_world,SOURCE
from kci.receipt_tasks import ReceiptTask
from kci.receipt_policy import graph_encode
from token_input_v3 import token_encode
timer=threading.Timer(9.5,lambda:os._exit(124));timer.daemon=True;timer.start();t=time.perf_counter();rows=[]
for depth,noise in ((1,0),(3,4),(1,16),(1,18),(1,30),(30,0)):
    env=ReceiptTask(make_world(900000+depth+noise,depth=depth,noise=noise),SOURCE)
    obs=env.observation(env.initial());action=env.actions(env.initial())[0];result={'depth':depth,'noise':noise}
    for name,encode in [('graph',graph_encode),('token',token_encode)]:
        try:encode(obs,action);result[name]='accepted'
        except ValueError as e:result[name]='explicit_error: '+str(e)
    rows.append(result)
assert rows[0]['graph']==rows[0]['token']=='accepted'
assert rows[3]['graph'].startswith('explicit_error:')
assert rows[4]['token'].startswith('explicit_error:')
assert rows[5]['token'].startswith('explicit_error:')
timer.cancel();report={'seconds':time.perf_counter()-t,'cases':rows,'no_silent_truncation':True,
    'not_a_generalization_result':True,'graph_message_passing_steps':6,'largest_evaluated_relation_chain':3,
    'large_scale_ready':False}
(R/'capacity_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
