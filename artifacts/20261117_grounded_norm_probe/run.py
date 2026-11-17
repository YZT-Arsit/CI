"""Frozen CPU clause probe; raw model responses sealed before label scoring."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def request(endpoint,payload=None,timeout=25):
    req=urllib.request.Request('http://127.0.0.1:11434/api/'+endpoint,
        data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def main():
    plan=json.loads((R/'PLAN.json').read_text())
    assert not (R/'RUN_PLAN.json').exists()
    assert all(sha(R/p)==h for p,h in plan['sources'].items())
    assert sha(R/'contexts.json')==plan['contexts_sha256']
    assert sha(R/'observations.jsonl')==plan['observations_sha256']
    assert sha(R.parent/'20261117_native_agent_probe/tokens.py')==plan['tokens_source_sha256']
    assert json.loads((R.parent/'20261117_optimization_repair/SEALED.json').read_text())['complete']
    observations=list(map(json.loads,(R/'observations.jsonl').read_text().splitlines()))
    assert len(observations)==plan['presentations']==84
    model=plan['model']['name']
    actual=next(row for row in request('tags')['models'] if row['name']==model)
    assert actual['digest']==plan['model']['digest']
    assert not request('ps')['models'], 'Existing loaded model: defer without changing it.'
    (R/'RUN_PLAN.json').write_text(json.dumps({'plan_sha256':sha(R/'PLAN.json'),'reads_labels':False,'retries':0,
        'model_digest':actual['digest'],'whole_goal_complete':False},indent=2))
    completed=0; failure=None
    try:
        start=time.perf_counter()
        warmup=request('generate',{'model':model,'prompt':'','stream':False,'keep_alive':'2m',
            'options':{**plan['options'],'num_predict':0}},60)
        (R/'WARMUP.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'response':warmup},indent=2))
        assert warmup.get('done') is True
        loaded=request('ps');(R/'LOADED_MODEL.json').write_text(json.dumps(loaded,indent=2))
        active=next(row for row in loaded['models'] if row['name']==model)
        assert active['digest']==actual['digest'] and active['size_vram']==0
        for row in observations:
            assert row['prompt_tokens']+plan['options']['num_predict']+64<=plan['options']['num_ctx']
            start=time.perf_counter()
            response=request('generate',{'model':model,'prompt':row['rendered_prompt'],'raw':True,'stream':False,
                'keep_alive':'2m','options':plan['options']},plan['request_timeout_seconds'])
            elapsed=time.perf_counter()-start
            with (R/'responses.jsonl').open('a') as stream:
                stream.write(json.dumps({'case_id':row['case_id'],'view':row['view'],'response':response,'seconds':elapsed})+'\n')
            completed+=1
            assert elapsed<=plan['request_timeout_seconds']
            assert response.get('prompt_eval_count')==row['prompt_tokens']
            assert response.get('done') is True and response.get('done_reason')!='length'
            if completed%12==0:print(json.dumps({'completed':completed,'seconds':elapsed}),flush=True)
    except Exception as error:
        failure={'type':type(error).__name__,'message':str(error),'completed':completed}
        (R/'FAILURE.json').write_text(json.dumps(failure,indent=2))
    finally:
        try:
            release=request('generate',{'model':model,'keep_alive':0},10)
            after=request('ps',timeout=5)
            (R/'RELEASE_CHECK.json').write_text(json.dumps({'response':release,
                'model_absent':all(row['name']!=model for row in after['models']),'after':after},indent=2))
        except Exception as error:
            (R/'RELEASE_FAILURE.json').write_text(json.dumps({'message':str(error),'fallback_keep_alive':'2m'}))
    files=['observations.jsonl']+(['responses.jsonl'] if (R/'responses.jsonl').exists() else [])
    (R/'RESPONSES_SEALED.json').write_text(json.dumps({'complete':completed==84 and failure is None,
        'completed_calls':completed,'failure':failure,'labels_scored':False,'files':{p:sha(R/p) for p in files}},indent=2))
    print(json.dumps({'completed':completed,'failure':failure}),flush=True)


if __name__=='__main__':
    main()
