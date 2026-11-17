"""Summarize completed checkpoints only; never turn a partial run into a result."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent

def main():
    stages=['query_pilot','actor_transfer','learned_evidence','coverage_repair','composition_repair_v2','risk_repair']
    checkpoints=[]
    for stage in stages:
        folder=R/stage
        assert (folder/'RESULTS.json').exists(),stage
        for training in sorted(folder.glob('*/training.jsonl')):
            arm=training.parent;sealpath=arm/('SEALED.json' if stage=='query_pilot' else 'CHECKPOINT_SEALED.json')
            seal=json.loads(sealpath.read_text());records=[json.loads(s) for s in training.read_text().splitlines()]
            expected=seal['actor_steps'] if stage=='actor_transfer' else seal['steps'];step_key='actor_step' if stage=='actor_transfer' else 'step'
            assert len(records)==expected and [r[step_key] for r in records]==list(range(1,expected+1))
            assert seal['adapter_hashes']
            for name,value in seal['adapter_hashes'].items():
                p=arm/'adapter'/name;assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==value
            checkpoints.append({'stage':stage,'arm':arm.name,'updates':len(records),'training_seconds_sum':sum(r['seconds'] for r in records),'max_update_seconds':max(r['seconds'] for r in records),'saved_seal':str(sealpath.relative_to(R)).replace('\\','/'),'token_units_not_comparable_across_stages':True})
    assert len(checkpoints)==12 and sum(r['updates'] for r in checkpoints)==5200
    audits={}
    for stage in ['learned_evidence','coverage_repair','composition_repair_v2','risk_repair']:
        audit=json.loads((R/stage/'AUDIT.json').read_text());assert audit['errors']==0;audits[stage]={k:v for k,v in audit.items() if k!='on_policy_query_accuracy_not_matched_across_arms'}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'single-seed CPU DEV repair; not final independent generalization evidence','checkpoints':checkpoints,'total_new_adapters':len(checkpoints),'total_optimizer_updates':sum(r['updates'] for r in checkpoints),'max_training_update_seconds':max(r['max_update_seconds'] for r in checkpoints),'audit_summaries':audits,'whole_goal_complete':False,'gpu_large_scale':'NO_GO','remaining':['multi-seed stability','multi-hop and structural generalization','safety-utility across all contract conditions','source-level task holdout','learned long-horizon role/message dependencies','natural norm applicability and cross-task validation']}
    result['alias_confirmation']=json.loads((R/'alias_confirmation/RESULTS.json').read_text());assert result['alias_confirmation']['errors']==0
    result['alias_token_length_check']=json.loads((R/'alias_confirmation/TOKEN_LENGTH_AUDIT.json').read_text())
    result['risk_format_attribution']=json.loads((R/'risk_repair/FORMAT_AUDIT.json').read_text())
    (R/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['\n## 7. 本轮最后两组对照的完成结果\n','下表均使用自由生成、无真值判断输入和无运行时动作屏蔽；配对成功要求两侧完整回合都安全完成。\n']
    for stage,title in [('composition_repair_v2','组合课程：每组 400 次更新'),('risk_repair','即时安全损失：每组 100 次更新')]:
        lines+=['### '+title+'\n','| 组别 | 条件 | 配对完整成功 | 配对 Planner 正确 |','|---|---|---:|---:|']
        for row in audits[stage]['paired']:lines.append('| {} | {} | {}/{} | {}/{} |'.format(row['arm'],row['condition'],row['both_strict_success'],row['groups'],row['both_planner_correct'],row['groups']))
        lines+=['','| 组别 | 非配对规范条件 | 完整成功 | 保持 CI 且格式有效 |','|---|---|---:|---:|']
        for row in json.loads((R/stage/'RESULTS.json').read_text())['summaries']:
            if row['condition'] in ['optional','authorized','draft','both']:lines.append('| {} | {} | {}/{} | {}/{} |'.format(row['arm'],row['condition'],row['strict_success'],row['episodes'],row['valid_ci'],row['episodes']))
        lines.append('')
    lines+=['本轮共保存 **12 个新 LoRA 检查点、5,200 次实际优化器更新**，均为种子 17。最长单次更新 {:.3f} 秒；每次更新计时包含循环内的提示构造、分词和训练，但不含启动、模型载入和评测，不能把它当作整个实验耗时。并非所有阶段都单独记录了载入耗时。原始训练与推理时长保留在日志中。'.format(result['max_training_update_seconds']), '', '这两组最终结果同样没有改变大批量 GPU 的 NO_GO 判断。所有数据为开发诊断；首次 GBK 执行失败、被取消的混杂方案和含捷径的控制版本均保留。']
    target=R/'REPORT.md';existing=target.read_text(encoding='utf-8');marker='\n## 7. 本轮最后两组对照的完成结果';target.write_text(existing.split(marker)[0]+'\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'adapters':12,'updates':5200,'max_update_seconds':result['max_training_update_seconds'],'gpu_large_scale':'NO_GO'}))

if __name__=='__main__':main()
