"""Synthetic distribution diagnostics; no LLM training or empirical privacy claim.

Each of six fixed workloads runs in a separate subprocess with a 10-second hard
timeout. Configuration is written before execution; raw sufficient statistics,
logs, receipts, source/configuration hashes, and a Chinese summary are retained.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time

CONFIG={
    'classification':'synthetic distribution diagnostic; NOT environment or LLM performance',
    'delta':0.05,'n_grid':[16,64,256,1024],'gap_grid':[0.0,0.05,0.15,0.30],
    'rare_violation_grid':[0.0001,0.001,0.01,0.05],
    'state_count_grid':[1,10,100,1000],
    'peek_n_grid':[2**k for k in range(4,17)],
    'seeds':{'null':2026111201,'rare':2026111202,'multiple':2026111203,
             'peeking':2026111204,'coverage':2026111205,'fixture':2026111206},
    'replicates':{'null':200000,'rare':200000,'multiple':5000,'peeking':100000,'coverage':20000},
    'multiple_n':256,'coverage_p_b':0.60,'illustrative_suffix_length':6,
    'fixture_n':256,'worker_timeout_seconds':10,
    'null_couplings':['independent','antithetic'],
    'predeclared_rule':'CI difference exceeds two-sided paired Hoeffding radius; no post-hoc threshold tuning',
}


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def radius(n,delta):return math.sqrt(2*math.log(4/delta)/n)


def exact_null_alpha(n,delta,coupling='antithetic'):
    trials=n if coupling=='antithetic' else 2*n
    scale=2 if coupling=='antithetic' else 1
    cutoff=n*radius(n,delta)
    return sum(math.exp(math.lgamma(trials+1)-math.lgamma(k+1)-math.lgamma(trials-k+1)
                        -trials*math.log(2)) for k in range(trials+1) if abs(scale*k-n)>cutoff)


def distributions(unit,out):
    import numpy as np
    rng=np.random.default_rng(CONFIG['seeds'][unit]);delta=CONFIG['delta']
    reps=CONFIG['replicates'][unit];rows=[];raw={}
    if unit=='null':
        for coupling in CONFIG['null_couplings']:
            for n in CONFIG['n_grid']:
                trials=n if coupling=='antithetic' else 2*n
                scale=2 if coupling=='antithetic' else 1
                counts=rng.binomial(trials,.5,size=reps)
                diff=scale*counts-n;cert=np.abs(diff)>n*radius(n,delta)
                raw[f'{coupling}_n{n}']=counts.astype(np.uint16)
                rows.append({'coupling':coupling,'n':n,'replicates':reps,
                             'false_certifications':int(cert.sum()),'rate':float(cert.mean()),
                             'exact_null_probability':exact_null_alpha(n,delta,coupling),
                             'hoeffding_radius':radius(n,delta)})
    elif unit=='rare':
        for q in CONFIG['rare_violation_grid']:
            for n in CONFIG['n_grid']:
                counts=rng.binomial(n,q,size=reps);miss=counts==0
                raw[f'q{q}_n{n}']=counts.astype(np.uint16)
                rows.append({'violation_probability':q,'n':n,'replicates':reps,
                             'zero_observed_violations':int(miss.sum()),'miss_rate':float(miss.mean()),
                             'exact_miss_probability':(1-q)**n,
                             'zero_observation_one_sided_95pct_upper_bound':1-delta**(1/n)})
    elif unit=='multiple':
        n=CONFIG['multiple_n'];single_alpha=exact_null_alpha(n,delta)
        for m in CONFIG['state_count_grid']:
            counts=rng.binomial(n,.5,size=(reps,m));diff=np.abs(2*counts-n)
            ordinary=(diff>n*radius(n,delta)).any(axis=1)
            adjusted=(diff>n*radius(n,delta/m)).any(axis=1)
            raw[f'm{m}_counts']=counts.astype(np.uint16)
            rows.append({'states':m,'n':n,'replicates':reps,
                         'family_false_certifications':int(ordinary.sum()),'family_rate':float(ordinary.mean()),
                         'exact_independent_family_rate':-math.expm1(m*math.log1p(-single_alpha)),
                         'bonferroni_false_certifications':int(adjusted.sum()),
                         'bonferroni_family_rate':float(adjusted.mean())})
    elif unit=='peeking':
        looks=CONFIG['peek_n_grid'];counts=np.zeros(reps,dtype=np.int64)
        seen=np.zeros(reps,dtype=bool);adjusted_seen=seen.copy();previous=0;history=[]
        for n in looks:
            counts+=rng.binomial(n-previous,.5,size=reps);history.append(counts.copy())
            crossing=np.abs(2*counts-n)>n*radius(n,delta)
            adjusted=np.abs(2*counts-n)>n*radius(n,delta/len(looks))
            first=crossing & ~seen;seen|=crossing;adjusted_seen|=adjusted
            rows.append({'n':n,'replicates':reps,'fixed_look_false_certifications':int(crossing.sum()),
                         'fixed_look_rate':float(crossing.mean()),'first_crossings':int(first.sum()),
                         'any_previous_crossing_count':int(seen.sum()),'any_previous_crossing_rate':float(seen.mean()),
                         'fixed_grid_bonferroni_any_crossing_count':int(adjusted_seen.sum()),
                         'fixed_grid_bonferroni_any_crossing_rate':float(adjusted_seen.mean())})
            previous=n
        raw['cumulative_binomial_counts']=np.stack(history,axis=1).astype(np.uint32)
    elif unit=='coverage':
        pb=CONFIG['coverage_p_b']
        for gap in CONFIG['gap_grid']:
            pa=pb+gap
            for n in CONFIG['n_grid']:
                ca=rng.binomial(n,pa,size=reps);cb=rng.binomial(n,pb,size=reps)
                diff=(ca-cb)/n;rad=radius(n,delta)
                raw[f'gap{gap}_n{n}_a']=ca.astype(np.uint16)
                raw[f'gap{gap}_n{n}_b']=cb.astype(np.uint16)
                rows.append({'p_a':pa,'p_b':pb,'true_ci_difference':gap,'n':n,'replicates':reps,
                             'prefer_a_count':int((diff>rad).sum()),'prefer_b_count':int((diff < -rad).sum()),
                             'inconclusive_count':int((np.abs(diff)<=rad).sum()),
                             'prefer_a_rate':float((diff>rad).mean()),'hoeffding_radius':rad,
                             'paired_continuation_calls':2*n,
                             'illustrative_action_samples':2*n*CONFIG['illustrative_suffix_length'],
                             'minimum_n_for_radius_below_true_gap':math.floor(2*math.log(4/delta)/gap**2)+1 if gap else None})
    np.savez_compressed(out/(unit+'_raw.npz'),**raw)
    return {'unit':unit,'seed':CONFIG['seeds'][unit],'classification':CONFIG['classification'],
            'raw_note':'Binomial sufficient statistics; exact generation and ordering are in the saved source.',
            'rows':rows}


def fixtures(out):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from kci import ToyCIEnv,SeededPolicy,PairedReferenceComparator
    from kci.contracts import Action
    env=ToyCIEnv();state=env.initial();a=env.act('A','SKIP');b=env.act('A','SEND','R',('identity',))
    def policy(obs,seed,unknown=False):
        local={f.id for f in obs.local_facts}
        if obs.actor=='A':return env.act('A','NOTIFY','Alice',answer='yes')
        if obs.actor=='R':
            return env.act('R','SEND','B',('rule',)) if obs.pending_request else env.act('R','FINALIZE',answer='yes')
        if 'eligible' in local:return env.act('B','SEND','R',('eligible',))
        if 'rule' in local:return env.act('B','COMPUTE_ELIGIBILITY')
        identity=any('identity' in m.fact_ids and m.recipient=='R' for m in obs.authorized_receipts)
        if unknown:
            return (Action('B','SEND','R',('band',),'unverified text') if seed%4==0
                    else env.act('B','SEND','R',('band',)))
        if identity:
            return env.act('B','SEND','R',('band',)) if seed%4==0 else env.act('B','REQUEST_RULES','R')
        return env.act('B','SEND','R',('band',)) if seed%4==0 else env.act('B','ABSTAIN')
    records={}
    for name,unknown in (('ci_priority',False),('known_violation_and_unknown',True)):
        frozen=SeededPolicy(lambda obs,seed:policy(obs,seed,unknown),f'statistical_fixture_{name}_v1')
        comparator=PairedReferenceComparator(env,frozen,n=CONFIG['fixture_n'],delta=CONFIG['delta'],
                                            seed_base=CONFIG['seeds']['fixture'])
        first=comparator.compare(state,a,b);second=comparator.compare(state,a,b)
        raw=asdict(first);repeat=asdict(second)
        h1=sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
        h2=sha256(json.dumps(repeat,sort_keys=True).encode()).hexdigest()
        assert h1==h2
        if unknown:
            assert first.preference is None and first.status=='unknown'
            assert 0<first.ci_known_pairs<first.n
        else:
            assert first.preference.chosen==a
            assert first.ci_diff_mean>first.ci_diff_radius and first.success_diff_mean<0
        records[name]={'environment_version':env.version,'reference_version':frozen.version,
                       'seed_base':comparator.seed_base,'same_seed_replay_equal':h1==h2,
                       'first_sha256':h1,'repeat_sha256':h2,'report':raw}
    return {'classification':'synthetic toy comparator integration fixtures; NOT LLM results','fixtures':records}


def summary(out,results,receipts):
    if len(results)!=6:
        text='# 统计压力诊断未完整完成\n\n请查看 receipts.json 中的超时或失败；不生成完整结论。\n'
    else:
        null=[r for r in results['null']['rows'] if r['coupling']=='antithetic' and r['n']==256][0]
        rare=[r for r in results['rare']['rows'] if r['violation_probability']==.001 and r['n']==64][0]
        multi=results['multiple']['rows'][-1];peek=results['peeking']['rows'][-1]
        lines=['# CPU 统计压力诊断','',
            '**证据等级：synthetic distribution diagnostic。** 除两个真实代码接口fixture外，数值来自预先固定的Bernoulli分布模拟，均不是环境性能或LLM实验。',
            '',f"六个工作单元均有独立10秒硬超时；最长实际子进程耗时 {max(r['elapsed_seconds'] for r in receipts):.3f} 秒。配置先于运行写入 config.json，全部随机种子、样本量、概率网格固定。",'',
            '## 已观察结果','',
            f"- 零效应、合法的反相关配对、n=256：错误认证 {null['false_certifications']}/{null['replicates']} = {null['rate']:.5%}；精确二项分布概率 {null['exact_null_probability']:.5%}。保守界降低误报，但不使误报概率为零。",
            f"- 真实违规率0.1%、n=64：未看到任何违规 {rare['zero_observed_violations']}/{rare['replicates']} = {rare['miss_rate']:.3%}；解析漏检概率 (1-q)^n = {rare['exact_miss_probability']:.3%}。零观测违规时95%单侧上界仍为 {rare['zero_observation_one_sided_95pct_upper_bound']:.3%}。",
            f"- 同时比较1000个独立零效应状态：至少一项误认证的比例 {multi['family_rate']:.3%}；使用每状态delta/1000后为 {multi['bonferroni_family_rate']:.3%}。这是合成独立状态下的family风险，不能直接套到相关真实状态。",
            f"- 对同一累积样本反复查看13个预定n（16至65536），未调整的任意时点越界率 {peek['any_previous_crossing_rate']:.3%}，只看最后一个固定时点为 {peek['fixed_look_rate']:.3%}；固定网格Bonferroni后为 {peek['fixed_grid_bonferroni_any_crossing_rate']:.3%}。这是optional-stopping风险演示；不把这些越界当成有效顺序证书。",
            '- 本轮未调整的任意时点越界率仍低于delta=0.05；观察到的是相对固定时点的概率膨胀，不声称本轮实证超过名义delta。',
            '- 真实comparator小fixture：CI较高但安全成功率较低的分支仍按CI优先选中；有完整成功见证。已知违规与UNKNOWN混合时返回unknown，不导出偏好。固定策略版本与种子重跑的完整报告SHA256一致。',
            '', '## 证据覆盖与调用成本','',
            '下面是纯分布中CI差值门槛的覆盖，不包含chosen必须有成功见证这一附加条件，因此不是完整训练数据覆盖率。p_B=0.60；动作调用估计假设每条续行6个模型动作，仅作计算量换算。','',
            '| 真CI差值 | n | 偏好A比例 | 配对续行调用数 | 假设动作采样数 |',
            '|---:|---:|---:|---:|---:|']
        for row in results['coverage']['rows']:
            lines.append(f"| {row['true_ci_difference']:.2f} | {row['n']} | {row['prefer_a_rate']:.3%} | {row['paired_continuation_calls']} | {row['illustrative_action_samples']} |")
        lines+=['','## 公式、限制与批量建议','',
            '- 配对差D∈[-1,1]；半径 epsilon=sqrt(2 log(4/delta)/n)，为CI与安全成功两个区间分配误差预算。即使CI在样本中相同，也不能证明总体等价。',
            '- 稀有违规漏检概率为(1-q)^n；零违规时单侧上界为1-delta^(1/n)。这不依赖模型是否自报安全。',
            '- 独立零效应状态的family误报概率为1-(1-alpha)^M；真实状态可能相关，实际alpha需独立校准，不能用独立公式作普遍保证。',
            '- 大批量硬偏好应预先分配整个family的误差预算（如每比较delta/M），并记录状态/动作对选择规则。FDR控制与零错误硬标签是不同承诺，不能混称。',
            '- 反复查看只能使用预定look网格及校正，或经过证明的alpha-spending/anytime-valid置信序列；不得每次复用固定样本delta并声称可选停止有效。',
            '- 候选选择、参考策略调试、动态增采样会带来额外选择偏差。本诊断使用固定概率与固定规则，不证明这些真实流水线的统计有效性。',
            '- comparator fixture采用确定性的seed分支以检验程序排序；不能将其重复运行当成独立随机模型的置信校准。',
            '- null独立配对与反相关配对均列入原始结果；反相关是允许的最不利配对之一，不意味着真实参考策略存在该相关结构。',
            '- 未训练模型、未调用外部服务；无方法优越性、真实隐私安全或泛化结论。','',
            '原始资料：config.json；各unit的JSON/NPZ；fixture.json；receipts.json；source_snapshot.py；manifest.json。NPZ保存二项充分统计量，结合冻结源代码可精确重放。']
        text='\n'.join(lines)+'\n'
    (out/'总结.md').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',choices=list(CONFIG['seeds']))
    parser.add_argument('--output-dir',type=Path);args=parser.parse_args()
    if args.worker:
        config=json.loads((args.output_dir/'config.json').read_text(encoding='utf-8'))
        assert config['configuration']==CONFIG
        result=fixtures(args.output_dir) if args.worker=='fixture' else distributions(args.worker,args.output_dir)
        write_json(args.output_dir/(args.worker+'.json'),result)
        print(json.dumps({'unit':args.worker,'status':'complete'},sort_keys=True));return
    base=Path(__file__).resolve().parents[2]/'artifacts'/'20261112_cpu_pilot'/'statistical_stress'
    out=args.output_dir or base/('run_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).read_bytes();(out/'source_snapshot.py').write_bytes(source)
    import numpy as np
    write_json(out/'config.json',{'configuration':CONFIG,'source_sha256':sha256(source).hexdigest(),
                                 'runtime':{'python':sys.version,'numpy':np.__version__,
                                            'random_generator':'numpy.default_rng / PCG64'},
                                 'frozen_before_execution_utc':datetime.now(timezone.utc).isoformat()})
    receipts=[];results={}
    for unit in CONFIG['seeds']:
        command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',unit,'--output-dir',str(out)]
        start=time.perf_counter();receipt={'unit':unit,'command':command,'timeout_seconds':10}
        try:
            run=subprocess.run(command,capture_output=True,text=True,timeout=10)
            receipt.update(returncode=run.returncode,timed_out=False)
            (out/(unit+'.log')).write_text(run.stdout+'\n'+run.stderr,encoding='utf-8')
            if run.returncode==0:results[unit]=json.loads((out/(unit+'.json')).read_text(encoding='utf-8'))
        except subprocess.TimeoutExpired:
            receipt.update(returncode=None,timed_out=True)
        receipt['elapsed_seconds']=time.perf_counter()-start;receipts.append(receipt)
        print(json.dumps(receipt,ensure_ascii=False),flush=True)
    write_json(out/'receipts.json',receipts);summary(out,results,receipts)
    write_json(out/'manifest.json',{p.name:sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print('ARTIFACT_DIRECTORY='+str(out));sys.exit(0 if len(results)==6 else 1)


if __name__=='__main__':main()
