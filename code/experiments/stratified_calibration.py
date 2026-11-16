"""Fixed finite-support statistical calibration; not environment validation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from kci.stratified import Trial, ComparisonPlan, FiniteRolloutPopulation


def run(output,seed):
    t=lambda ci,s,c:Trial(ci,s,c)
    populations={
        'null':[(t(True,True,4),t(True,True,4))]*16,
        'success':[(t(True,True,4),t(True,i<4,4)) for i in range(16)],
        'cost':[(t(True,True,3+i%2),t(True,True,12+i%2)) for i in range(16)],
        'risk_priority':[(t(i<8,True,2),t(True,True,12)) for i in range(16)],
        'tiny_risk_blocks_cost':[(t(i!=0,True,2),t(True,True,12)) for i in range(1024)],
        'unknown':[(t(True,True,2),t(None,None,4))],
        'conditional_cost_unresolved':[(t(True,i<8,2),t(True,i>=8,12)) for i in range(16)],
    }
    results=[]
    for name,pairs in populations.items():
        population=FiniteRolloutPopulation(pairs,{'calibration':name,'reference':'uniform_support_v1'})
        for certificate in (False,True):
            report=population.compare(ComparisonPlan(seed=seed),use_upper_certificate=certificate)
            report.update(case=name,certificate_enabled=certificate)
            results.append(report)
    Path(output).write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(dict(Counter((r['case']+':'+str(r['certificate_enabled'])+':'+str(r['level'])) for r in results))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();run(args.output,args.seed)
