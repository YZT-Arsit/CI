"""Independently recheck one family of saved matched ledger pairs per worker."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code'))
from kci.contracts import Fact
from kci.training_data import PinnedTokenizerAssets,TrainingSequenceAdapter
from multi_env_independent_audit import expected_inferences

def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True);a=p.parse_args();start=time.perf_counter()
    base=ROOT/'artifacts/20261113_iteration2/environment/history_matched_v2'
    manifest=ROOT/'artifacts/20261112_cpu_pilot/tokenization/config.json'
    adapter=TrainingSequenceAdapter(PinnedTokenizerAssets.from_manifest(manifest,
        expected_manifest_sha256='e1242b65a53ea00e4747af4af876f7c867ed53510e80ff0f8edc4f3911012d21'))
    results=[]
    for path in sorted((base/'cases').glob(a.family+'_*.json')):
        doc=json.loads(path.read_text(encoding='utf-8'));left,right=doc['records']
        assert left['chosen']==right['rejected'] and left['rejected']==right['chosen']
        prompts=[json.loads(r['prompt']) for r in (left,right)]
        assert [k for k in prompts[0] if prompts[0][k]!=prompts[1][k]]==['authorized_delivered_messages']
        messages=[p['authorized_delivered_messages'] for p in prompts]
        assert len(messages[0])==len(messages[1])==doc['spec']['count']+1
        changed=[i for i,(x,y) in enumerate(zip(*messages)) if x!=y];assert len(changed)==1
        index=changed[0];expected_index={'front':0,'middle':doc['spec']['count']//2,'end':doc['spec']['count']}[doc['spec']['position']]
        assert index==expected_index
        assert [k for k in messages[0][index] if messages[0][index][k]!=messages[1][index][k]]==['body']
        assert messages[0][index]['event_id']=='evt1' and messages[0][index]['sender']=='A'
        encodings=[adapter.encode_pair(r) for r in (left,right)]
        assert len(left['prompt'])==len(right['prompt'])
        assert len(encodings[0].prompt_ids)==len(encodings[1].prompt_ids)
        assert encodings[0].prompt_ids!=encodings[1].prompt_ids
        inferred=[]
        for row,side,encoded in zip((left,right),doc['sides'],encodings):
            cfg=row['metadata']['fixture_configuration']['base_configuration']
            facts={k:Fact(**v) for k,v in cfg['facts'].items()}
            env=SimpleNamespace(family=cfg['family'],facts=facts,context_ids=tuple(sorted(k for k in facts if k.startswith('context_'))))
            found=expected_inferences(env,set(env.context_ids)|{'payload_0'})
            inferred.append(any(v.startswith('protected:') for v in found))
            assert len(encoded.prompt_ids)==side['prompt_tokens']
            assert encoded.required_tokens==side['required_pair_tokens']
        assert inferred==[True,False]
        assert doc['sides'][0]['raw_action_json']==doc['sides'][1]['raw_action_json']
        assert [s['raw_verdict'] for s in doc['sides']]==['violation','valid']
        results.append({'case':doc['spec']['case_id'],'tokens':len(encodings[0].prompt_ids),'closure_flip':True,
            'prompt_only_context_body_changes':True,'case_sha256':sha256(path.read_bytes()).hexdigest()})
    assert len(results)==20
    summary={'family':a.family,'cases_rechecked':len(results),'passed':True,'cases':results,
      'scope':'independent declared-closure and exact-token/visible-input control; no model forward or human validation',
      'limits':['same synthetic workflow; fixed related/unrelated lexical values may remain a shortcut',
                'external heartbeat ledger is an initial condition, not an executed long agent trajectory'],
      'seconds':time.perf_counter()-start,'audit_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    out=ROOT/'artifacts/20261113_iteration2/research/matched_history_audit';out.mkdir(exist_ok=True)
    (out/(a.family+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='cases'}))

if __name__=='__main__':main()
