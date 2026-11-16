"""Build a source-bound summary of this bounded CPU pilot, not a paper result."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
read=lambda p:json.loads((ROOT/p).read_text(encoding='utf-8'))
dataset=read('dataset/dataset_summary.json')
model=read('ollama_agent/summary.json')
replay=read('ollama_agent/replay_audit_final.json')
optimizer=read('optimizer/summary.json')
assert optimizer['completed']==7 and all(optimizer['checks'].values())
tokenization=read('tokenization/summary.json')
assert tokenization['status']=='PASS' and tokenization['shared_prompt_verified_pairs']==1824
causal_boundary=read('tokenization/causal_boundary.json')
assert causal_boundary['history_causes_ci_label_flip_in_declared_toy']
test_text=(ROOT/'job_logs/final_full_regression.stderr.txt').read_text()
tests=int(re.search(r'Ran (\d+) tests',test_text).group(1))
assert test_text.rstrip().endswith('OK')
factors={}
for name in ('knowledge','history_length','search_budget','horizons_costs','reference_policies','statistics','semantic_faults'):
    folder='factors_v2' if name in {'horizons_costs','reference_policies'} else 'factors'
    factors[name]=read(f'{folder}/{name}.json')
assert all(x['cases']==x['passed'] for x in factors.values())
proxies={};proxy_count=0
for folder in ('preference_proxy','preference_proxy_v2','order_probe_v2','balance_probe'):
    assert read(folder+'_final_verification.json')['all_pass']
    groups=defaultdict(list)
    for path in sorted((ROOT/folder).glob('*.json')):
        row=json.loads(path.read_text());groups[row['variant']].append(row);proxy_count+=1
    for variant,rows in groups.items():
        proxies[folder+'/'+variant]={
            'runs':len(rows),'mean_pair_fit':statistics.mean(r['after']['overall'] for r in rows),
            'safe_success':sum(r['trajectory_diagnostic']['safe_successes'] for r in rows),
            'violations':sum(r['trajectory_diagnostic']['confirmed_violations'] for r in rows),
            'episodes':sum(r['trajectory_diagnostic']['episodes'] for r in rows),
            'ci_pair_fit':statistics.mean(r['after']['by_source']['ci'] for r in rows),
            'future_pair_fit':statistics.mean(r['after']['by_source']['future'] for r in rows),
            'max_gradient_error':max(r['gradient_check_absolute_error'] for r in rows)}
search_status=Counter(s for row in factors['search_budget']['rows'] for s in row['statuses'])
safe_costs=[r['cost'] for r in model['results'] if r['status']=='SAFE_SUCCESS']
mean_step=model['total_worker_seconds']/model['total_decision_requests']
readiness={
    'research_objective_changed':False,'bounded_cpu_engineering':'GO_WITHIN_DECLARED_TOY_SCOPE',
    'bulk_synthetic_replication':'DO_NOT_TREAT_AS_MORE_INDEPENDENT_DATA',
    'bulk_llm_training':'NO_GO','method_effectiveness':'NOT_ESTABLISHED',
    'gates':{'software_regression':True,'synthetic_record_replay':True,'local_model_adapter_smoke':True,
             'independent_base_worlds_and_holdout':False,'natural_language_semantic_validation':False,
             'independent_dynamic_ci_norm_labels':False,'reference_statistical_batch_error_control':False,
             'real_llm_training_and_matched_ablations':False,
             'cpu_token_sequence_precheck':True,'finite_optimizer_invariants':True},
    'gpu_deferred':['4B-8B shared LoRA/SFT/DPO','actual model precision and batch forward/backward equivalence',
                    'multi-seed matched full-model ablations','batched frozen-reference inference'],
    'cpu_first':['independent worlds and rule families','observable-state/label consistency',
                 'human-calibrated derived-fact authorization','free-text verifier and unknown routing',
                 'predeclared batch-level statistical error and cost budget',
                 'integrate checked sequence contract into the actual trainer; preserve long-history supervision']}
(ROOT/'readiness.json').write_text(json.dumps(readiness,ensure_ascii=False,indent=2),encoding='utf-8')
summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'tests':tests,
         'factor_cases':sum(x['cases'] for x in factors.values()),'dataset':dataset,
         'proxy_policies':proxy_count,'proxies':proxies,'reference_search_status_counts':dict(search_status),
         'llm':{k:model[k] for k in ('model','episode_counts','total_episodes','total_decision_requests',
                                   'completed_responses','max_worker_seconds','total_worker_seconds',
                                   'prompt_eval_tokens','generated_tokens','dpr_verified_repairs')},
         'mean_llm_worker_seconds':mean_step,'optimizer_diagnostics':optimizer,
         'tokenization_diagnostics':tokenization,'causal_truncation_diagnostic':causal_boundary,
         'readiness':readiness}
(ROOT/'results_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')

def table_rows(keys):
    return '\n'.join(f"| {label} | {proxies[key]['mean_pair_fit']:.1%} | {proxies[key]['safe_success']}/{proxies[key]['episodes']} | {proxies[key]['violations']} |" for key,label in keys)

proxy_table=table_rows([
 ('preference_proxy_v2/sft_only','仅合成轨迹 SFT'),
 ('preference_proxy_v2/ci_only_sft','SFT → CI 偏好'),
 ('preference_proxy_v2/ci_future_sft','SFT → CI+future'),
 ('preference_proxy_v2/all_sft','SFT → CI+future+repair'),
 ('balance_probe/all_sft_ci_x4','上述混合，CI采样权重×4'),
 ('balance_probe/all_sft_ci_x14','上述混合，CI采样权重×14')])
order_rows='\n'.join(f"| {label} | {proxies['order_probe_v2/'+key]['ci_pair_fit']:.1%} | {proxies['order_probe_v2/'+key]['future_pair_fit']:.1%} |" for key,label in [('ci_then_future','CI→future'),('future_then_ci','future→CI'),('balanced_ci_future','等量混合顺序')])
report=f'''# KCI-Align 修复与 CPU 初步实验报告

## Material Passport

日期：2026-11-12，America/Los_Angeles。模式：实现修复、有限模拟与结果验证。研究目标：**接收者知识实际积累条件下，共享执行模型的即时 CI 与长期合规任务策略学习**。目标未改变。

证据分为原始本地论文、合成结构化环境、有限线性训练代理、冻结真实 LLM 菜单执行四类，互不替代。原代码/文档备份见 `baseline/`，全部输入 SHA256 见 `input_manifest.json`。原 Proposal、Handoff、SPEC 和29篇PDF未修改。

## 当前结论

**工程原型可以继续小规模、有边界地验证；目前不应启动大批量 LLM 训练，也没有建立方法优越性。** 阻碍首先是数据独立性、语义/规范真值、可观测性和参考证据覆盖，GPU只是后续一部分。

本轮完成 {tests}/{tests} 回归测试、{summary['factor_cases']} 个因素检查、2,016条合成偏好的逐条重放、{proxy_count}个有限策略训练/顺序/配比诊断，以及真实Qwen的12段受限执行。所有正式实验worker设置≤10秒上限；快速检查不人为凑时长。模型主pilot最长worker {model['max_worker_seconds']:.3f}秒，2次未解决请求保留UNKNOWN。没有启动GPU训练、下载新模型权重或调用外部模型服务。Ollama模型已停止驻留、磁盘权重保留，`ollama_cleanup.json`确认无驻留模型。

## 1. 文献如何约束后续研究

29篇PDF共756页均已本地提取；其中9篇重点阅读指定78页，其余20篇仅摘要/开篇筛查。提取不是全文精读，未逐图核验。详见[文献审计](literature/literature_audit.md)及页码/哈希矩阵。

- PrivAct已有候选树、终局评分回传及同Prompt DPO；这套骨架不能单独作为新意。
- TG-CI已有接收者披露状态及未来传播约束；PrivacyFlow已有背景知识和推断后果规划。
- CDI已有首次失败位置训练，但更新外部instructor且其反馈范围不同；DPR需证明完整后果验证及执行模型更新带来的额外收益。
- ConCI支持相关变化flip、无关变化hold及联合评分；Privacy Collapse提示更新后需单独测CI，但其论文SFT证据不能直接推广成DPO结论。

保留的研究增量候选仍是：**两个当前均合法行动对未来合规完成能力的不同影响，能否通过可信监督被执行模型学到，并超越匹配预算的普通即时/轨迹训练。** 本轮没有回答这个方法效果问题。

## 2. 已完成的修复

| 部分 | 具体结果 |
|---|---|
| 知识与观察 | 纳入本地背景和授权已交付日志；删除任务中的身份提示；规范/成本变化反映在观察中 |
| 行动与消息 | 未知角色不再崩溃；非消息字段受限；请求、决策、通知有交付事件；决策metadata不泄漏oracle正确性 |
| 证据身份 | 同Prompt、角色、参考/生成器版本、预算、期限校验；环境配置哈希与State成本进入身份 |
| 缓存/并发 | 输入、缓存及输出见证隔离；8线程32次检查保持4/7成本，修改一个输出不污染其他结果 |
| DPR | 未失败前缀和UNKNOWN不制造repair；合法上游路径优化仍单列CCE |
| 统计 | 已知违规正确推出safe-success=False；有限样本不证明高优先级等价；统计成本差仅作诊断 |
| 终局与解析 | 空成功suffix保留真实terminal state；JSON数组/null/坏结构计为拒绝，不崩溃 |
| 数据 | 严格事件字段、版本、见证、来源hash；同Prompt反向严格偏好拒绝并保留冲突证据 |

新增38项回归，加原24项共62项通过；原测试中因背景知识和统计合同纠正而需要的预期/fixture变动有记录。最终demo仍为4/7成本及B处REQUEST_RULES修正。实现细节见[核心](core_changes.md)、[环境](env_changes.md)、[批量完整性](batch_integrity/总结.md)和[实现补记](../../docs/KCI_Align_CPU_Implementation_Amendment.md)。

一个新发现尤为关键：R拥有不同隐藏背景时，B的可见Prompt可能完全相同，却得到相反正确标签。导出门槛可阻止当前批次的矛盾对，但**不能让模型凭不可见信息学会区分**。真实环境仍需审查合法可见状态、查询机会或不确定决策。

## 3. 建立的数据集及其限制

| 项目 | 数量/状态 |
|---|---:|
| 原始偏好记录 | 2,016 |
| 最终core逐条重放通过 | 2,016 / 2,016 |
| 去除同Prompt同动作对的重复后 | 1,824 |
| 姓名/编号/同为eligible的薪资带归一后 | 38种偏好模式 |
| 表面换名配置 | 48 |
| 独立基础任务 / 推断规则家族 | 1 / 1 |
| 任务答案 | 全部eligible=yes |
| 独立validation/test基础世界 | 0 / 0 |

文件：[完整偏好](dataset/preferences.jsonl)、[38种代理输入](dataset/canonical_preferences.jsonl)、[数据卡](dataset/DATASET_CARD.md)、[汇总](dataset/dataset_summary.json)。每条保留环境配置、可见Prompt、前缀行动、审计、参考见证、版本及来源。12个shard的最终verification同时绑定输入SHA256、core SHA256和校验脚本SHA256。

原始来源计数为CI=96、future=1,344、repair=576；唯一pair中CI∩repair=96、future∩repair=96。**全部CI pair也属于repair；不能把重复加权解释成DPR独立增益。** 38种canonical patterns中CI/repair重叠2种、future/repair重叠2种、repair-only8种、future-only26种。

这些记录全部是新建的合成数据，不是PiSAs动态真值，也不是独立人工核验集。主分组将48个换名版本归为同一基础组；取消原计划中的独立holdout评价，协议补记保留原因。数据适合程序回归和管线诊断，尚不适合正式训练/泛化论证。

## 4. 因素模拟与批量风险

知识/授权/相似编号/草稿/重复记录96项，历史长度10项，预算24项，期限与非单位成本18项，手写参考策略20项，统计接口12项，语义/动作故障7项，共187项通过各自声明的检查。历史长度测试只是0–256条受控无关记录下oracle标签和运行成本检查，不是模型长上下文能力结果。参考策略行是手写控制，不能当训练模型。

预算网格的48个分支证据计数为 `{dict(search_status)}`；覆盖不足时保持unknown/witness，未把搜索失败当不可达。词面改写不能通过当前精确语义核验时仍返回UNKNOWN；这说明自由文本验证器尚缺，而非已解决paraphrase。

统计压力诊断明确使用合成分布，结果不是实际Agent误差率：

- 真违规概率0.1%、n=64时，完全漏检率93.822%（解析约93.797%）。
- n=256反相关零效应配对，单比较误认证约0.320%；1000个独立比较至少一次误认证为96.020%；分配总体误差预算后约0.140%。
- 13次累积查看越界率3.326%，末次固定查看0.319%；本次仍低于名义δ=0.05，只支持相对膨胀，不能声称突破δ。
- 真CI差0.05、n=1024时，偏好A阈值检出率仅约2.15%（430/20,000）；尚不包含成功见证条件，也不是置信区间覆盖率。小效果可能需要很高参考预算。

配置、原始计数和复现见[统计诊断](statistical_stress/总结.md)。当前统计认证器仍是单对接口，尚未自动落实全训练集的错误预算。

## 5. 真实本地模型初步执行

模型为Qwen2.5-7B-Instruct Q4_K_M，digest记录在runtime与模型产物中。固定温度0、种子、菜单、规范和期限；实际size_vram=0，context=4096。执行者只选菜单index，实际行动与正文由受限环境执行；未提供参考最优路径或其最优成本，单步行动成本正常可见。

| 主pilot结果 | 数量 |
|---|---:|
| 合规任务成功 | 4 |
| 任务失败 | 6 |
| 未解决请求/轨迹 | 2 |
| 确认CI违规 | 0 |
| 总episode | 12 |

1个失败起点在期限6内本来就不可达；剩余11个可行起点中，4成功、5可避免失败、2未知。全部62个请求输入、60个执行动作在最终core下重放一致。5个可避免失败获得DPR修正见证，但**这些修正尚未用于真实模型训练**，不能说模型已学会修复。

已成功轨迹平均成本为{statistics.mean(safe_costs):.2f}单位。常见问题是合法时仍走请求规则/本地计算长路线，期限不足时漏掉最终通知。0次确认违规不证明低违规概率，尤其样本少、存在未知、菜单受限且全部同模板。

主pilot的60个已完成请求累计报告{model['prompt_eval_tokens']:,}个prompt tokens、{model['generated_tokens']}个生成tokens；两次中断的token成本未知。62个worker合计{model['total_worker_seconds']:.1f}秒，均值{mean_step:.2f}秒/请求。作为**条件外推而非实测规模结果**，若每状态2分支×64种子×7后续步，按这个均值约需{2*64*7*mean_step/3600:.2f} CPU小时/状态；实际将受缓存、批处理、轨迹长度和硬超时影响。这提示未来采用LLM参考续行时可能出现的数据生成瓶颈；本轮合成偏好由符号搜索生成，未实测该规模瓶颈。

独立8请求菜单反序检查中，4对条件都选同一语义REQUEST_RULES；仅排除了这些观察的固定index解释。较早静态probe的不同编号合法项被误判DENY，和主pilot的保守路由值得后续检查。静态future问题直接给4/7成本，只是选择sanity，不能视为自主长期规划。

完整报告见[真实模型pilot](ollama_agent/REPORT.md)、[最终重放](ollama_agent/replay_audit_final.json)。原始输出、超时和错误均保留，未重试挑选更好的回答。

## 6. CPU训练代理：拟合与轨迹必须分开

这是280参数有限线性菜单策略，使用人工结构语义特征；不是LLM。全部主要代理的标准DPO梯度有限差分最大误差为{max(v['max_gradient_error'] for v in proxies.values()):.4g}。π0冻结在随机初始或SFT初始，实际更新共享参数。所有评价均同基础世界、同语料，**没有独立泛化评价**。

以下合计3个种子×12起点；每种子11个可行起点，因此全部36轨迹中可行者33个。分母同时保留原36，未丢弃困难/不可达例。

| 初始化与DPO数据 | 同语料偏好拟合 | 合规成功 / 36 | 确认违规次数 |
|---|---:|---:|---:|
{proxy_table}

CI只有2/38种模式。默认混合可在约95%拟合时仍把这两类CI模式判错；调整采样配比后，该小代理可减少这类违规，但×14是探索中的敏感性设置，**不是推荐的大模型参数，也不是方法增益证据**。SFT初始化本身提供了更多任务步骤覆盖，故不能把随机初始与SFT结果差异归因于DPO。

随机初始的beta=10代理甚至能拟合38/38偏好，仍仅12/36轨迹成功；偏好差分矩阵rank仅16，未监督菜单动作和任务步骤仍欠约束。这解释了为什么loss、pair accuracy与整轨迹成功必须分开。

额外等source预算顺序诊断复用了相同CI/future minibatch多重集，仅改变顺序（不含repair）：

| 顺序 | 训练后CI pair正确率 | future pair正确率 |
|---|---:|---:|
{order_rows}

另保存了10%/30%标签翻转、缺少关联历史/规范特征、repair×10和beta敏感性结果。实际翻转为4/38与11/38；`no_history`只移除显式关联标志，仍保留任务进度等观察，不是完整历史删除。beta固定学习率同时改变有效更新幅度；这些探索不能用来做方法排名。全部66份主要代理权重已在最终core下重放一致。

### 6.1 放大训练前的CPU优化器诊断

另完成7组NumPy数学诊断，最长worker为{optimizer['max_worker_seconds']:.3f}秒；沿用同一38×280设计矩阵，不增加任务或训练证据。正确的样本数加权microbatch累积与整batch最大差约1.39e-17，模拟不等长rank用全局计数归一的差约2.78e-17。直接平均不等长批次/角色/rank均值、先逐microbatch裁剪、每步刷新reference都给出不同结果。

显式Adam在48步中的第19步保存并恢复参数、m/v、步数和RNG，最终参数与抽样序列逐位相同；丢弃Adam状态或重置RNG得到非零差异。这将断点恢复和实验配对落实为可检查合同。FP16/32/64及极端margin只验证NumPy算术，稳定logaddexp避免本例朴素exp溢出；**没有验证真实BF16、DDP进程组或大模型内核**。详见[优化器专项](optimizer/REPORT.md)。

### 6.2 真实tokenizer与截断检查

使用官方Qwen2.5-7B-Instruct tokenizer，固定revision `{tokenization['official_revision']}`及文件SHA256，依赖安装在本轮目录内。没有下载模型权重或改全局Python环境。1,824/1,824去重pair的chosen/rejected prompt token前缀一致，3,648个回复分支均监督EOS；18个主worker最长{tokenization['max_worker_seconds']:.3f}秒，加两项辅助检查共20个worker、最长{tokenization['max_experiment_worker_seconds']:.3f}秒。原始2,016对全部映射到这1,824对，另外192对为重复。

64种序列在左右padding、batch 1/2/8/32、不同PAD设置下共1,024次合成logprob汇总一致；PAD复用EOS时按token ID直接屏蔽会误删真实EOS。这个测试检查mask、position IDs和移位计分，不是模型forward。序列求和与按长度取平均有排序反例，继续保留原DPO条件序列log-likelihood求和目标。

这批数据的prompt为252–335 tokens。max_length=256会丢失全部回复EOS并拒绝全部1,824对；512及以上均可完整保留。独立的因果控制先在toy环境确认：A真实交付目标身份后，同一个B band行动由VALID变VIOLATION；再把相关记录置于15,362-token控制前部，左截断到256–4,096均会删去它。**这只是规范因果控制加文本截断检查，没有测模型截断后的违规行为，也没有把残缺文本重判为合法Observation。** 较早的15,374-token控制只证明丢失普通交付记录，其非因果边界仍保留。

当前预处理对不完整prompt/回复/EOS采取整对拒绝。批量训练还必须报告拒绝率和分布变化，不能通过丢掉长历史样本把本研究变成短上下文任务。后续需充足上下文预算或另经验证的保真压缩。HF与本地GGUF的special IDs吻合、一个无tools模板实例吻合，但模板源码不完全相同且未逐项比较vocab/merges，未建立完整tokenizer等价性。详见[预处理专项](tokenization/REPORT.md)。

## 7. 是否投入大批量实验

| 门槛 | 当前判断 |
|---|---|
| 有限环境程序与证据回归 | 通过，限定当前声明域 |
| 可重放合成偏好管线 | 通过，限定同一合成任务 |
| 本地真实模型接入 | 可运行，但效用和时限仍有问题 |
| 独立任务、规则家族及严格留出 | 未通过：独立基础世界只有1个 |
| 自由文本语义、派生事实授权真值 | 未通过：目前是精确结构语义与手工规范 |
| 合法观察足以决定标签 | 存在明确反例，尚需环境层解决 |
| 批量统计误差与参考成本预算 | 尚未落实到真实生成服务 |
| CCE/DPR对真实LLM的独立训练增益 | 未验证 |

因此：**可以继续小批量环境/标签建设；暂不投入大批量LLM训练。** 下一阶段先构造多个真正独立的基础任务和规则家族，形成可重放、可见信息一致、有人类/独立证据支持的动态规范标签；再校准候选覆盖与参考策略，冻结真实训练/评测划分及误差预算。完成后才进行4B–8B共享LoRA的SFT/DPO、小规模即时/长期/repair匹配消融。

本轮已在CPU检查真实tokenizer及mask/EOS/padding/截断的有限合同；GPU阶段再验证实际大模型的批大小、数值精度、优化器/参考快照、scaler/scheduler、共享角色梯度及训后真实行为。同worker内的NumPy恢复不能代替真实框架跨进程恢复，有限策略代理不能替代大模型前向/反向等价性检查。WSL环境本轮未改动。

## 8. 交付、复现与停止理由

- `REPORT.md`、`results_summary.json`、`readiness.json`：总报告与机器可读结论。
- `dataset/`：原始shard、全量重放凭据、合并偏好、结构代理输入、来源重叠及独立性审计。
- `literature/`：29篇来源hash、逐页文本、9篇重点阅读记录和独立实验审查。
- `ollama_agent/`、`ollama/`：冻结真实模型的原始请求/响应、执行轨迹、未知记录和离线重放。
- `statistical_stress/`、`batch_integrity/`、`factors*/`：分布压力、缺陷前后证据和因素读数。
- `optimizer/`、`tokenization/`及[因素覆盖表](scale_factor_coverage.md)：批次、优化器、序列与数值合同及仍未覆盖的规模化因素。
- `preference_proxy*/`、`order_probe_v2/`、`balance_probe/`：小代理参数、固定参考、梯度校验及结果；历史探索保留。
- manifests、receipts、job_logs和最终文件hash：复现命令、版本、每单元时限与过程错误。早期一次回归因版本只读属性与测试fixture冲突失败，已修复并保留日志；未删掉失败证据。
- `changes.patch`、`changed_files.json`、`FINAL_VALIDATION.json`、`final_manifest.json`：代码差异、原始资料未改检查、证据与最终core绑定检查、交付文件哈希。隔离依赖文件不逐个列入最终清单，其wheel来源和哈希单列在tokenization目录。

复现核心：从 `code/` 执行 `python -B -m unittest discover -s tests -v`。批处理入口为 `python -B -m experiments.run_short_jobs <新manifest路径>`；重复实验请新建输出目录和manifest。修改数据或core后必须重新生成绑定hash的verification，旧verified状态不可沿用。

本轮在研究目标不变的前提下完成了修复和多方面短探索。继续复制同一toy或重复小代理拟合，不能补足独立任务、规范真值或真实LLM训练证据，因此停止扩量并交付这些已确认的边界。这里没有将未完成的研究宣布成功，也没有将方向改成仅运行时过滤器。
'''
(ROOT/'REPORT.md').write_text(report,encoding='utf-8')
card=f'''# KCI CPU synthetic v1 数据卡

用途：有限环境回归、证据链/训练管线诊断。禁止据此宣称真实CI、独立世界泛化或方法收益。

来源：本项目原创建模数据，无PiSAs/MultiAgentPrivacy样本复制，无人工规范真值。2016条原始记录，1824条唯一表面对，38种canonical偏好模式。48组名称/编号/eligible薪资带只是同一基础世界的变体；基础组统一toy_eligibility_join_v1，规则家族identity_attribute_join_v1。没有独立validation/test。

`preferences.jsonl`保留prompt/chosen/rejected及环境配置、实际前缀、直接/增量审计、参考见证、版本、来源成员、原始文件行位置和哈希。`canonical_preferences.jsonl`是38种结构代理输入，保留对应原记录ID与重放配方，不是独立任务集合。`shards/*.verification.v2.json`绑定原始字节SHA256与最终core；失配会阻止assemble。

来源计数：CI96、future1344、repair576。CI与repair重叠96，future与repair重叠96；不能把交叉成员当额外独立样本。最终全局精确同Prompt反向pair扫描为0；另有隐藏背景反例被保存在batch_integrity并隔离，说明零冲突不保证所有未来数据都可观测。

规则限制：精确事实ID、canonical消息、单identity-attribute join、固定eligible=yes。全部human_verified=false、natural_language_semantics_verified=false。许可/治理：这里仅声称原创建模；引用论文PDF不是本数据的标签授权或隐私证明。

生成代码：code/experiments/build_dataset.py；合并代码：assemble_dataset.py；独立结构审计：corpus_audit.py。完整配置和版本见相邻summary及上层input_manifest和final_manifest。
'''
(ROOT/'dataset/DATASET_CARD.md').write_text(card,encoding='utf-8')
print(json.dumps({'tests':tests,'factor_cases':summary['factor_cases'],'proxy_policies':proxy_count,
                  'bulk_training':readiness['bulk_llm_training'],'report':str(ROOT/'REPORT.md')},ensure_ascii=False))
