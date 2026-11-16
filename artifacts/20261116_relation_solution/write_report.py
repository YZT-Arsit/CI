"""Assemble the final Chinese report from sealed counts, never from log impressions."""
import json
from pathlib import Path
R=Path(__file__).resolve().parent

def load(path):return json.loads((R/path).read_text(encoding='utf-8'))
def ratio(record,key='safe_success'):return f"{record[key]}/{record['n']}"

def run():
    results=load('RESULTS.json');table=[];proofs={}
    forced=load('FORCED_DIAGNOSTIC.json');assert forced['states_checked']==3840 and not forced['errors']
    pairbase=[r for r in load('PAIRED_DECISION_DIAGNOSTIC.json')['records'] if r['condition']=='test']
    pairvalid=sum(r['both_planner_valid'] for r in pairbase);pairsame=sum(r['same_route_when_both_valid'] for r in pairbase);paircorrect=sum(r['both_planner_optimal'] for r in pairbase)
    forced_table=[]
    for arm in ('local','plain','trace','balanced'):
        for role in ('Source','Worker'):
            rows=[r for r in forced['summaries'] if r['model'].startswith(arm+'_') and r['role']==role]
            assert len(rows)==3
            n=sum(r['states'] for r in rows);correct=sum(r['correct'] for r in rows);unsafe=sum(r['unsafe_valid'] for r in rows);lost=sum(r['safe_but_lost_completion'] for r in rows);invalid=n-sum(r['valid'] for r in rows)
            forced_table.append(f'| {arm} / {role} | {correct}/{n} | {unsafe} | {lost} | {invalid} |')
    for folder in ('language_evaluation','balanced_language_evaluation'):
        audit=load('execution_audits/'+folder+'_SUMMARY.json')
        for model in audit['models']:proofs[model['model']]=model['proofs']
    for arm in ('plain','trace','balanced'):
        folder='balanced_language_evaluation' if arm=='balanced' else 'language_evaluation'
        for seed in (17,29,41):
            name=f'{arm}_{seed}';data={r['condition']:r for r in results['evaluations'][folder] if r['model']==name};paired={r['condition']:r for r in results['models'][folder+'/'+name]['paired']}
            p=proofs[name];faithful=f"{p['faithful']}/{p['total']}" if p['total'] else '不生成证明'
            external=next(r for r in results['evaluations']['balanced_language_evaluation' if arm=='balanced' else 'language_external_complete'] if r['model']==name and r['condition']=='external_complete')
            native_folder=R/'language_native_transfer'/name
            native_left={r['world_id']:r for r in map(json.loads,(native_folder/'native_base.jsonl').read_text(encoding='utf-8').splitlines())};native_right={r['world_id']:r for r in map(json.loads,(native_folder/'native_swapped.jsonl').read_text(encoding='utf-8').splitlines())};assert native_left.keys()==native_right.keys()
            native_pair=f"{sum(native_left[k]['safe_success'] and native_right[k]['safe_success'] for k in native_left)}/{len(native_left)}"
            table.append(f"| {name} | {paired['test']['both_success']}/{paired['test']['groups']} | {paired['depth10']['both_success']}/{paired['depth10']['groups']} | {ratio(data['authorized'])} | {ratio(data['both'],'valid_ci')} | {ratio(external)} | {native_pair} | {faithful} |")
    replay=[]
    for p in sorted((R/'execution_audits').glob('*_SUMMARY.json')):
        d=json.loads(p.read_text(encoding='utf-8'));replay.append((p.stem,d['episodes'],d['errors']))
    rawcounts=sum(n for _,n,_ in replay);errors=sum(e for _,_,e in replay)
    native=load('structured_native_transfer/EVALUATION_SEALED.json')['summaries'];native_success=sum(r['safe_success'] for r in native);native_n=sum(r['episodes'] for r in native)
    trained=[]
    for path in ('language_models/CHECKPOINTS_SEALED.json','balanced_language_models/CHECKPOINTS_SEALED.json'):trained+=load(path)['models']
    maxstep=max(max(x['seconds'] for x in load(('balanced_language_models' if m['model'].startswith('balanced') else 'language_models')+'/'+m['model']+'/training.json')) for m in trained)
    tokenbudget={m['model']:m['target_tokens'] for m in trained}
    timing=load('time_budget_probe/RESULTS.json')['summaries'];old_invalid=sum(r['old_invalid'] for r in timing);recovered=sum(r['recovered_format'] for r in timing);recovered_correct=sum(r['recovered_and_optimal'] for r in timing)
    text=f'''# 三小时修复报告：关系泛化、安全与任务效用

结论：已经得到一套通过本轮完整验证的结构化参考方案；共享预训练语言模型仍未通过关系学习门槛。**当前不进入 GPU 大批量实验。** 原研究目标没有改成“用规则代替语言模型”。

## 方案是什么，为什么这样修

采用 **外部任务数据 + 自建配对干预 + 分开的安全/可完成性监督 + 局部关系学习**。
完整通俗说明见 [SOLUTION.md](SOLUTION.md)，条件性推导见 [MECHANISM_SCOPE.md](MECHANISM_SCOPE.md)。

例如，甲路线的必需输出能接上接收者已有的私人历史，乙不能接上，应选乙并完成任务。
只交换历史连接对象，正确选择就必须反转。授权、允许删字段、历史尚未发送以及两条路线均无解时，
答案还要相应改变。动作数量、智能体流程成本与截止时间都不能充当答案线索。

本轮针对五个具体问题进行了修复与对照：

1. **Source 看不到未来任务依赖。** 新编码保留所选路线必须提交的关联字段，并与已送达事实分开。
2. **安全监督导致拒绝捷径。** 为每个状态的每个动作提供实际执行后的安全与合规可完成性标签；平局保留为平局。
3. **长链传播失效。** 增加共享、可重复的关系组合模型，区分结构先验与实际学到的参数。
4. **符号表示误匹配。** 16 维代码在大干扰下误把不同实体关联起来；固定权重、提高到 64 维并明确扩容后，用新种子确认修复。
5. **语言训练被格式词元稀释。** 比较只答动作、生成证明、以及动作/安全/任务/路径/格式分项加权三种训练。分项加权仍未通过门槛，所有失败保留。

## 数据集选择与实际建立的数据

| 数据 | 本轮实际范围 | 能支持什么 |
|---|---:|---|
| 自建训练 | 576 基础组，1,152 配对世界，5,760 状态，13,824 动作标签 | 精确控制关系、授权、任务义务与历史送达 |
| 开发集 | 96 世界，480 状态 | 发现平局虚高、错误证明与标签失衡 |
| 主关系测试 | 13 分层各 64 世界，另有 4 组等量历史交换 | 符号、深度、噪声、方向、断边、环、分支及安全—效用条件 |
| 外部计算 | 原有 Meeting/Query；新增 PiSAs Meeting 10–17，共 8 个来源实例 | 验证真实业务约束与计算结果，不是独立 CI 规范标签 |
| 新增压力/确认集 | 最高 32 跳、128 条干扰；多套相互隔离的新种子 | 发现表示与容量问题，避免在同一失败样例上修完自评 |

训练数据的全部保存标签通过独立 BFS 实现复核，错误为 0；训练基础组与开发/测试组没有交叉。
8,480 组配对状态/词池比较的 token 长度、实体词表与动作顺序全部一致，历史反转没有引入提示长度线索。
新增外部任务起初仅 1/8 可执行，补齐“房间功能”和“会议须在指定时间前结束”后，8/8 的计算结果与原始日程一致。
原不兼容记录保留。会议业务截止时间不是智能体的人工流程长度限制。

**这些合成世界不等于相同数量的独立真实任务。** 模型读的是统一结构化关系，外部迁移主要验证工具计算与集成；
新增 CI 历史和规范仍是合成的，尚不能宣称自然语言跨领域 CI 泛化或独立人类规范验证。
“新符号”在这里主要指实体 ID/名称；I/L/H 等关系类型的语义与规则家族保持固定，尚未证明新规范谓词或自由文本规范解析的泛化。

## 结构化模型：哪些修复确实有效

| 对照/修复 | 观察到的结果 |
|---|---|
| 不变的旧 V5 | 10 跳安全完成各 24/64；部分种子在 4 跳即失败 |
| 相近图架构 + 新密集标签 | 短链可完成；10 跳为 0/64、24/64、24/64，数据改进不能消除固定传播深度问题 |
| 局部关系监督 + 16 维表示 | 主门槛三种子通过；仍有少量误拒绝，10 跳配对为 63/64、62/64、62/64 |
| 固定权重改为 64 维 | 最高 32 跳、128 干扰的组合测试中 576/576 安全完成；对应 16 维为 391/576 |
| 修复后完整条件复验 | 新的 3,360 回合全部符合各自验收标准：可行任务完成，无解任务保持安全；三种子均通过 |
| 去掉关系绑定能力 | 基础与长链任务均无法安全完成，不能靠动作/长度特征替代关系 |
| 只学安全、打乱监督标签 | 出现拒绝/效用崩塌；安全比例高不等于会完成任务 |

原生输出对照也复验了：**{native_success}/{native_n} 安全完成**，完整结果没有添加新的凭据字段。
该对照仍使用合成初始知识、规范与改造过的候选披露协议，不能冒称原论文的自然 CI 标签或完整原生智能体复现。
部分额外诊断曾因原三小时预算暂停；用户明确允许适度延长后，恢复并完成语言原生输出迁移与强制中间状态诊断。
早先的暂停记录保留为历史。结构模型完成 2,400 个强制状态，语言模型完成 1,440 个强制状态；均与完整策略回合分开报告。

可复用接口是 `code/kci/relation_runtime.py::choose_action`，使用 64 维实体表示、256 事实上限；
没有运行时闭包答案、动作屏蔽或规则替换。576 个确认轨迹经接口重算，动作与执行记录完全一致。

这个稳定版本接受了局部关系门和相等判断监督，组合运算结构是提供的归纳偏置。
动作评分固定为“4 × 预测合规概率 + 预测可完成性概率”，并未按测试成绩搜索这个系数；
它不提供形式安全保证，也不参与执行器的价值标签定义。
标签打乱控制破坏角色/动作内的监督对应关系，也影响规范条件，不能解读成只干预关系的纯单因素消融。
补充的直接端到端训练不使用局部教师：独立动作采样及配对分层采样都只有两个种子稳定，另一个退化成拒绝。
所以没有挑选成功种子宣布“无需辅助监督”。后者关键 Worker 条件的期望采样比例由约 1.95% 提高到 5.56%，
仍不足以保证三种子稳定；相关负结果见 `auxiliary_ablation`、`direct_pair_repair` 和对应完整复验。

## 真实预训练语言模型：不能用结构结果替代

本轮实际训练 SmolLM2-135M-Instruct，CPU FP32，LoRA q/v，460,800 可训练参数；
三个训练方案、每方案三个种子、每个检查点固定 600 更新。`balanced` 在唯一最优 DEV 诊断后冻结，
没有根据主测试结果选择检查点。训练目标包括 EOS，排除提示词；证明/标签仅在训练目标中出现。
最长实际训练更新为 {maxstep:.3f} 秒。下载/安装/加载单列为准备开销。

表中配对成功要求同一基础世界的两种历史都安全完成。无解列要求有效输出且合规，格式失败不能拿安全分。
证明可信要求所有候选的安全/任务标签正确，而且每一条声称的路径边确实存在；仅答案碰巧对不算。

| 模型/种子 | 基础配对 | 10 跳配对 | 获授权后完成 | 无解时有效合规 | 外部任务完成 | 原生输出配对 | 可信证明/生成次数 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

这些语言模型没有达到原研究目标的门槛。原始未适配模型在同样短答案格式预算下全部格式失败，
不能把微调后学会格式等同于学会关系。早期 joint pilot 的 45/48 DEV 分数主要被平局抬高，已明确撤销其作为关系证据的资格。
进一步检查第一次路线决策：在九个检查点的基础测试中，两侧均可解析的 {pairvalid} 组里，
有 {pairsame} 组在交换历史后仍选同一路线；只有 {paircorrect} 组两侧路线都选对。
这直接显示关键历史关系尚未可靠影响路线选择，而不只是后续任务执行失败。
逐模型数据见 `PAIRED_DECISION_DIAGNOSTIC.json`；这里仍是重复模型诊断计数，不是独立现实任务数量。

固定中间状态的诊断进一步分离了错误位置。下表汇总三个种子的回合数用于定位错误，
不把它们当成独立样本估计总体置信度；“正确”要求立即合规，并在可行时保留完成机会。

| 方案 / 决策角色 | 正确 | 有效但不安全 | 安全但丢失完成机会 | 格式/时限失败 |
|---|---:|---:|---:|---:|
{chr(10).join(forced_table)}

Source 的许多状态可用同一个 Deliver 动作处理，其高分本身不能证明关系推理。
Worker 才必须根据关系与许可区分 Full、Minimal、Stop；这里的错误说明问题不只是 Planner 的路线选择。
所有 3,840 个强制状态标签及原始答案解析均经独立 BFS 逻辑复核，详见 `FORCED_DIAGNOSTIC.json`。

主测试保持原设定：纯答案最多 16 个新 token，证明方案最多 256 个新 token，每个生成批次上限约 8 秒。
另在明确选定的受限失败批次上，将时限提高到 12 秒，其他输入、批次成员和 256-token 上限保持不变：
原本格式失败的 {old_invalid} 个决策中，{recovered} 个恢复可解析格式，其中 {recovered_correct} 个同时选对动作。
这是输出时限敏感性诊断，不回写主成绩；`time_budget_probe` 保存原输出、新输出及前缀一致性检查。
该诊断只放宽时间，没有排除 256-token 上限导致的截断；因此不能把所有长链格式失败都解释为关系能力失败。
强制 Worker 状态中有效但不安全、或有效但丢失完成机会的输出，提供了独立于格式失败的行为证据。

证明目标比纯答案目标包含更多训练词元，分项加权还改变了损失尺度，所以不能据此声称完整 KCI-DPO 方法收益或严格等计算量胜出。
本轮 SFT 是可学习性诊断；135M/600 更新的失败也不能证明更大模型或更充分训练必然失败。

## 大规模实验前暴露的问题

| 因素 | 本轮处理/观察 |
|---|---|
| 无关信息增加 | 16 维近似匹配误拒绝随干扰增多，64 维修复在新集确认 |
| 输入容量 | 超容量明确失败，不静默截断；语言词池 128 个实体，64 条干扰时已可能超限 |
| 深度外推 | 固定 6 次传播与可重复组合呈明显差别；过程没有步数奖励捷径 |
| 稀有条件与类别平衡 | 仅平衡标签仍可能忽略“无许可但可安全完整发布”等组合；已记录两种采样的精确覆盖 |
| 安全/效用冲突 | 不用泄露率单独验收；同时报告任务完成、误拒绝、无解安全和强制危险中间状态 |
| 证明幻觉 | 保存模型原始证明，输出封存后逐边核验，不把教师答案传回推理 |
| 规模复杂度 | 结构模型使用稠密匹配和重复传播，当前性能不构成无限规模保证；未进行 GPU/分布式训练 |

## 验证、交付与下一步

180 项回归通过，其中历史 tokenizer 测试在其原 Windows 环境运行；WSL 的不兼容导入记录保留。
从 Git 暂存快照恢复后的新增 7 项测试再次通过，测试本身耗时 7.230 秒；含 WSL 启动/导入共 21.007 秒。
首次 9.5 秒整段启动超时及续跑记录均保留，训练和主评测预算没有改变。
已完成的标准轨迹独立重放共 {rawcounts:,} 回合，执行差异 {errors}；各模型、来源、数据、压缩文件均保留哈希。
小样本门槛是工程点估计，不是总体可靠率至少 95% 的统计证明。不同种子和同一世界的变体没有被当成独立现实任务。

当前决定：**结构化研究基线可继续使用；共享语言模型关系学习未过关；GPU 大批量 NO-GO。**
下一步应新建并封存测试组，先解决语言模型的局部实体匹配、单步关系与安全/可完成性判断，
再组合成长链；同时补独立真实 CI 规范验证。通过这些小门槛后，才值得进行有限 GPU 对比，最后才是大批量训练。
本轮没有启动任何 GPU 任务，也没有把 SFT 或结构模型冒称为原 KCI-DPO 方法已证明有效。

主要入口：[机器结果](RESULTS.json)、[逐项结果表](RESULTS.csv)、[数据审计](DATA_AUDIT.json)、
[复跑说明](REPRODUCE.md)、[训练/测试预设](PLAN.md)、[语言训练预设](LANGUAGE_PLAN.md)。
JSONL 以可逐字节核验的 gzip 归档，先运行 `python -B tools/restore_relation_artifacts.py`。

参考来源：[ProofWriter](https://aclanthology.org/2021.findings-acl.317/)、
[CLRS](https://github.com/google-deepmind/clrs)、
[PiSAs 固定版本](https://huggingface.co/datasets/ServiceNow/PiSAs/tree/ac6634013dc74b1c9601d9e59248a59465438a43)、
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)。前两者用于方法设计参考，本轮未用其数据训练。
'''
    (R/'REPORT.md').write_text(text,encoding='utf-8');(R/'LANGUAGE_TOKEN_BUDGET.json').write_text(json.dumps(tokenbudget,indent=2));print('REPORT.md generated from sealed results')

if __name__=='__main__':run()
