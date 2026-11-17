"""Independent label/selection checks for completed frozen text interventions."""
import json,sys,hashlib,re
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[1];OLD=R.parent/'20261116_relation_solution';sys.path[:0]=[str(OLD),str(ROOT/'code')]
from audit_data import check
from kci.action_binding_metrics import pair_metrics

def read(path):return list(map(json.loads,path.read_text(encoding='utf-8').splitlines()))

def run():
    cases=read(R/'cases.jsonl');assert not [error for row in cases for error in check(row)]
    lookup={(r['pair_id'],r['side']):r for r in cases};seal=json.loads((R/'PROBE_SEALED.json').read_text());assert all(r['identity_max_probability_difference']<1e-6 for r in seal['identity_controls'])
    records={};total=0
    for report in seal['models']:
        path=R/(report['model']+'.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==report['outputs_sha256'];rs=read(path);assert len(rs)==144
        for row in rs:
            case=lookup[row['pair_id'],row['side']];assert row['optimal_action']==case['optimal_action']
            assert row['actions']==[c['action'] for c in case['candidates']] or set(row['actions'])=={c['action'] for c in case['candidates']}
            top=row['full_vocab_top_token'];selected=row['actions'][ord(top)-65] if top in ('A','B') else None
            assert row['selected']==selected and abs(sum(row['legal_probabilities'])-row['legal_mass'])<1e-6
            records[report['model'],row['pair_id'],row['side'],row['mode']]=row;total+=1
    components=json.loads((R/'components/RESULTS.json').read_text());component_rows=0
    for summary in components['summaries']:
        path=R/'components'/(summary['model']+'_'+summary['mode']+'.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==summary['raw_sha256'];rs=read(path)
        actual=pair_metrics(rs);assert all(summary[k]==v for k,v in actual.items())
        for row in rs:
            key=row['model'],row['pair_id'],row['side'];gold=records[*key,'gold_trace'];other=records[*key,'swapped_trace'];prefix=gold['assistant_prefix'];mode=row['mode']
            assert row['prompt']==gold['prompt'] and row['actions']==gold['actions'] and row['optimal_action']==gold['optimal_action']
            if mode=='task_flipped':expected=re.sub(r'TASK ([01])',lambda m:'TASK '+str(1-int(m[1])),prefix)
            elif mode=='task_zero':expected=re.sub(r'TASK [01]','TASK 0',prefix)
            elif mode=='path_erased':expected=re.sub(r'PATH [^;\n]+;','PATH none;',prefix)
            else:
                paths=iter(re.findall(r'PATH ([^;\n]+);',other['assistant_prefix']));expected=re.sub(r'PATH [^;\n]+;',lambda m:'PATH '+next(paths)+';',prefix)
            assert row['assistant_prefix']==expected;component_rows+=1
    result={'cases_independently_checked':len(cases),'frozen_readouts_checked':total,'component_interventions_checked':component_rows,'errors':0,'scope':'saved labels, selected token mapping and targeted text mutations; not an independent human review or proof of internal causal abstraction'}
    (R/'DIAGNOSTIC_AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':run()
