from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,re
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding='utf-8'))
def write(n,x):(R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
legacy=read('legacy_signal_summary.json');data=read('dynamic_data_v3/summary.json')
finite=read('finite_audit.json');qwen=read('qwen_audit.json');replay=read('data_replay_audit.json')
obs=read('observation_audit.json');sequence=read('sequence_v3/summary.json')
cce=[read('audit_cce_'+mode+'.json') for mode in ('safe','careless','mixed')]
counts=Counter()
for d in cce:counts.update(d['counts'])
certified=sum(v for k,v in counts.items() if 'STATISTICAL_PREFERENCE' in k)
unknown=sum(v for k,v in counts.items() if '_UNKNOWN_' in k)
tests=int(re.search(r'Ran (\d+) tests',(R/'job_logs/final_124_tests.stderr.txt').read_text(encoding='utf-8')).group(1))
assert tests==124 and certified==74 and unknown==118
assert all(d['all_passed'] for d in cce+[finite,qwen,replay])
arms='\n'.join(f"| {label} | {finite['arms'][arm]['pair_correct']['heldout_horizon']}/12 | {finite['arms'][arm]['episodes'].get('SAFE_SUCCESS',0)}/48 | {finite['arms'][arm]['episodes'].get('VIOLATION',0)} | {finite['arms'][arm]['episodes'].get('TASK_FAILURE',0)} |" for arm,label in [('sft_only','共同SFT基线'),('ci','SFT + 即时CI'),('history','SFT + 历史主监督'),('ordinary','SFT + 其他状态future')])
gate={'synthetic_history_signal':'GO_WITH_EXPLICIT_FINITE_CONTRACT',
      'small_real_llm_training':'CONDITIONAL_GO_FOR_ENGINEERING_PILOT_ONLY',
      'conditions':['implement and verify actual Trainer/model forward and masked sequence loss',
                    'allocate GPU and verify checkpoint/reference/optimizer/numerical equivalence',
                    'retain synthetic norm and two-source-instance scope; no independent task claim'],
      'large_method_experiments':'NO_GO','real_llm_learning_effect':'NOT_ESTABLISHED',
      'arbitrary_black_box_CCE_lower_tiers_without_proof':'UNKNOWN',
      'independent_task_holdout':False,'human_norm_verification':False,'objective_changed':False}
requirements=[
 {'requirement':'1 future分类与历史CI监督','evidence':['legacy_signal_summary.json','dynamic_data_v3/history_pairs.jsonl','training_v3/history_primary.jsonl','data_replay_audit.json'],'status':'bounded_requirement_verified','limits':'8 primary pairs, 4 matched groups, 2 source instances; not scale or natural-language norm proof'},
 {'requirement':'2 分层非穷举CCE与交叉校准','evidence':['../../code/kci/stream_cce.py','dynamic_cce_v3','audit_cce_safe.json','audit_cce_careless.json','audit_cce_mixed.json'],'status':'bounded_requirement_verified','limits':'strict lower tiers require verified upper equivalence; arbitrary black-box references without proof remain UNKNOWN'},
 {'requirement':'3 两个动态任务合同','evidence':['TASK_CONTRACTS.md','audit_task_meeting.json','audit_task_query.json'],'status':'bounded_requirement_verified','limits':'different task graphs, shared three-relation privacy grammar; synthetic adaptation, not faithful reproduction'},
 {'requirement':'4 可见性与标签隔离','evidence':['observation_audit.json','deprived_quarantine.jsonl','dynamic_data_v3/quarantine.jsonl'],'status':'bounded_requirement_verified','limits':'enumerated support only; no global hidden-state identifiability guarantee'},
 {'requirement':'5 有限模型/Qwen/DPR诊断','evidence':['finite_audit.json','qwen_audit.json','data_replay_audit.json'],'status':'bounded_requirement_verified','limits':'Qwen is next-action only plus separately labelled hybrid suffix; no LLM training; DPR unique increment 0'}]
write('readiness.json',gate);write('completion_audit.json',requirements)
write('results_summary.json',{'generated_utc':datetime.now(timezone.utc).isoformat(),'tests':tests,
   'legacy':legacy,'data':data,'finite':finite,'qwen':qwen,'cce_counts':dict(counts),'cce_certified':certified,
   'cce_unknown':unknown,'replay':replay,'readiness':gate,'requirements':requirements})
report=f'''# KCI-Align 第三阶段：长期 CI 信号与训练准备

## Material Passport

2026-11-13 08:39 UTC 开始，约一小时授权窗口。Proposal v0.6 的研究问题、共享角色执行模型和 CI → 合规完成 → 成本优先级保持不变。所有单项实验/request worker 上限≤9.5秒；快速检查不人为延长。旧轮次产物只读。本轮没有GPU训练或外部模型服务。

**结论：已建立小而可重放的合成长期 CI 主监督，有限模型出现初步可学习信号；真实 LLM 学习收益仍未建立。** 可进入有条件的GPU工程试跑准备，不能进入大批量方法实验。这里的“通过”分别指软件合同、明确合成规范下的数据和有限诊断，不指真实社会规范或论文效果。

## 1. 旧 future 数据实际包含什么

将112个原始配置按相同合法prefix/action重放，仅放开受保护派生关系的授权，任务、动作、期限不变；再检查严格偏好是否仍成立。全部112短worker通过，最长0.310秒，发布唯一pair的贡献来源分类一致。

| 类别 | 原有唯一future偏好 |
|---|---:|
| 普通任务可达性 | 3926 |
| 普通合规路径成本 | 226 |
| CI约束介导的可达性 | 12 |
| CI约束介导的路径成本 | 12 |
| 总计 | 4176 |

只有24条在移除CI约束后失去原严格优势。**授权干预不是历史干预，因此该审计证明的历史翻转数量仍为0。** 没有把4176条都称为长期知识状态学习信号。逐条分类：[legacy_signal_categories.jsonl](legacy_signal_categories.jsonl)。

## 2. 新增两个动态任务，明确保留改造边界

[任务合同](TASK_CONTRACTS.md)说明角色权限、实际交付、工具执行、规范、任务oracle及许可来源。

| 合同 | 实际任务与不同的流程机制 | 历史对后续价值的影响 |
|---|---|---|
| Meeting | PiSAs固定实例的时段/房间/人员约束；私有预约ACK、HR清理流程、实际预约及Client通知 | 先前case-cause与当前合法roster、未来case-link组合成受限实名医疗事实 |
| Query | MultiAgentPrivacy benign run_1的公司/站点/城市任务；分片任意顺序交付、投影、join、Vault结果转发 | 先前company-owner-name与当前合法owner-index、未来地址bundle组合成受限实名地址 |

这两个任务有不同的状态图、工具依赖、任务计算和终局，但**共用三关系join隐私语法**，不是两个新的推断家族。Meeting保留原始业务约束；Query保留原始benign任务，不使用有阈值歧义的sensitive run_2。别名、动态权限、保护关系、历史控制、工具接口和成本均为明确的合成改造。原子敏感片段的许可与原benchmark全局标签不同，不能称faithful隐私复现。

每个动作成本为实际执行API步数，包含历史和最终通知，不是人工给每种历史加奖励：

| 任务 | 无关历史：公开/私有路线成本 | 相关历史：公开/私有路线成本 | 短期限效果 |
|---|---|---|---|
| Meeting | 5 / 6 | 7 / 6 | horizon6：相关公开路线无合规成功续行 |
| Query | 6 / 7 | 8 / 7 | horizon7：相关公开路线无合规成功续行 |

长预算分别9/10时，两条当前行动都合规且能成功，最优成本方向随一个真实交付的历史记录反转；授权派生关系后反转消失。四组配对的事件数、字符数及完整Qwen chat prompt token数一致（Meeting362，Query473）。仍可能存在固定键/词汇捷径；没有跨新名称/新源实例泛化证据。

## 3. 发布数据与可辨识性

最终版本：[dynamic_data_v3](dynamic_data_v3/summary.json)，训练格式：[training_v3](training_v3/summary.json)。

- **8条历史翻转主监督**，对应4组配对、2个源实例、2个改造任务合同；其中4条长预算样本用于有限模型训练，4条短预算样本只作期限留出诊断。
- 全部辅助与主偏好合计208条：42成本、162可达性、4即时CI。其余200条不作为新增历史因果信号计算。16个参数配置和374个枚举状态不是16/374个独立任务。
- 默认合法观察下，374个Prompt各自唯一，范围内无冲突隔离项；该门槛不能证明范围外隐藏世界的可辨识性。
- 移除授权回执的诊断产生10个冲突观察类、32个不一致严格方向，保存在[deprived_quarantine.jsonl](deprived_quarantine.jsonl)，未进入训练。主配对在该干预下具有相同可见输入和相反正确方向。
- 执行输入只有协议、角色、合法路由键、授权交付/ACK、规范与预算。任务结果通过真实工具回执交付；reference值、CI真值和oracle正确性位均不进入Prompt。

全部208条通过现有严格Observation/Action适配器和真实固定tokenizer，完整序列394–967 tokens，4k保留208/208，EOS/mask/padding合同通过。新增 `dynamic_training.execute_tool_action` 拒绝错误角色和非约定body/recipient，避免把模型改写的正文悄悄忽略后执行另一条消息。仍只支持固定工具动作，非自由文本CI。

## 4. CCE修复：严格分层、上层证据和UNKNOWN

新入口 `kci/stream_cce.py::compare_rollouts` 对冻结参考策略实际续行的成对结果，分别构造CI、合规成功、归一成本三个同时区间；固定n、δ、成本上界及state/actions/reference绑定。旧比较器保留其原有合同，新任务校准使用新入口。

**有限采样不能证明完全相等的高优先级风险。** 因此无上层等价证明时，不因“没看到违规”转而认证成功率或成本优势。已验证的有限参考支持集可提供上层证书；证书绑定源/代码/状态/动作/策略，漂移拒绝。成本层当前要求所有支持续行均合规成功；混合成功条件成本仍UNKNOWN。未采用事后容差、奖励加权或把小风险换取低成本。

三个观察权限内的冻结参考控制：safe、careless、mixed。mixed只有一个贯穿整条续行的随机二值选择，其余步骤确定。每个支持分支实际执行并保存轨迹；重复随机抽样可复用同一绑定下的确定结果。这是小型参考控制，**不是Qwen参考策略的完整分布证明**。一般黑箱LLM缺少这样的证书时，下层比较继续UNKNOWN。

最终192项校准=16配置×3策略×2样本量(256/8192)×有/无证书；预声明总δ=.01，每项δ=.01/192，每项内部三个同时Hoeffding区间，单位动作成本界16。PRNG抽样独立性和审计正确性是前提；不是anytime保证。

| 校准结果 | 数量 |
|---|---:|
| 统计偏好 | 74：32 CI、4成功率、38成本 |
| UNKNOWN | 118 |
| 与完整有限参考分布的严格方向矛盾 | 0/74 |

特别是safe参考下，上层全合规/同成功有证据时，大n认证14个成本比较和2个成功率比较；小n只认证2个成功率比较，其余成本UNKNOWN。无上层证明的同合规成本比较全部UNKNOWN。参考分布值与有限任务最优值分别保存，不将坏参考策略的失败叫作无解。

初始112个纯结果表校准是单项δ=.01，**不宣称它们具有合并family保证**。最终192项为另一个明确计划。v1/v2/v3为固定协议修复重放，不累计成独立样本量；v3与v2的决策全部一致。

## 5. 有限共享模型：出现信号，尚不足以证明方法收益

12个有限词特征模型：4臂×3种子(17/29/41)，共享线性参数，只有可见Prompt的词/相邻词hash特征，无闭包、风险位或正确答案特征。共同200步SFT排除初始路线决策，再做固定200步DPO、batch8、lr=.3、β=1；SFT-only不做DPO。每个种子的reference权重逐位一致，独立标量差分梯度误差最大约1.78e-11。

history臂只使用4条长预算主偏好；CI臂4条，其他状态future臂118条。相同步数不意味着相同独立数据量。代码中的`ordinary`只是该对照的历史名称，准确含义是“非初始状态future”，未将其再次因果审计为纯普通任务。

| 配置 | 短期限配对正确 | 合规成功 | 违规 | 任务失败 |
|---|---:|---:|---:|---:|
{arms}

history臂训练期限配对9/12，短期限10/12；SFT基线均6/12。每臂48回合=16参数配置×3种子，不是48个独立任务。history仍有5次违规；3种子不足以建立稳定优势或显著性，期限留出也不是任务留出。它仅支持“这8条合成监督包含有限表示可利用的条件差异”的初步判断。

## 6. 冻结Qwen与DPR

最终Qwen2.5-7B Q4_K_M固定参数、CPU、4096 context：16个初始路线决策+4个下游决策，20个请求全部完成，19合法动作、1违规，精确偏好匹配13/20。初始12/16匹配，但四组无授权历史配对的两侧**同时正确为0/4**；下游仅1/4为最优。当前合法动作本身不能说明未来选择正确。

这是**每case只调用一次模型的定点诊断**；后续由冻结safe参考执行，只用于确定后果，不能报告为完整Qwen回合成功率。3个决策存在离线可行修复；没有拿修复去训练或重试Qwen。最终20请求约58.52秒，最长6.039秒，已完成token7192输入/40生成。因去掉早期历史长度差异，本轮保留两个完整版本：共40个task请求、2次固定warmup；正式统计仅v2的20个，不把重放计作新独立样本。驻留模型已释放。

枚举数据中的DPR潜在修复状态158个，逐条执行失败/修复及成功见证重放通过，其中即时CI违规修复3个。**158个repair pair全部与主/辅助偏好重叠，独立新增pair为0。** 它们是新合同内的修复覆盖，不是158个独立研究任务，也不是DPR额外学习收益。

## 7. 修复、负证据与最终核验

- 上层精确等价证书必须优先于采样噪声。构造性非IID不平衡控制暴露了初版会给出与证书矛盾的CI偏好，已修正；该控制不是观测到的实际随机误判率。反例和原文件保留。
- 初版state_id没有包含完整合同；最终绑定source、规范授权、期限、任务和状态。case_id此前随记录保存，未发现标签因此混用。
- Query无关历史由四位ID改为三位不相交ID，清除明显字符长度线索；重新生成数据、完整重跑有限模型和20请求Qwen，原版本保留。
- 第一次重放审计把内存tuple与JSON list直接比较，产生检查器失败；规范化JSON后复核通过，原失败日志和检查器保留。
- 最终 **124/124测试通过**。另一种DFS和关系连接实现复算374状态/606行动；任务终局分别对照固定源oracle与原benign答案。192 CCE结果的样本均值、区间、支持轨迹重新核算；192有限模型回合、20Qwen动作与混合后果、158修复见证、208完整编码均重放通过。

这是同一研究执行者的第二实现与自检，不是外部人类独立审阅。原始版本、失败和边界不会因最终检查通过而被删除。

## 8. Go / No-Go 与GPU待办

| 层次 | 判断 |
|---|---|
| 精确合成合同中的长期主信号 | GO：8条、4配对、2任务；限制已写明 |
| 小规模真实LLM工程试跑输入 | 条件GO：结构/完整token/工具接入已准备；Trainer与真实模型前后向仍待接入验证 |
| 任意黑箱参考下的下层CCE | 无可验证上层证据时UNKNOWN；不能宣称通用问题已消失 |
| 真实LLM方法学习效果 | 未建立；冻结Qwen不是训练结果 |
| 大规模方法实验/论文收益结论 | NO-GO：2源实例、无独立任务留出、合成规范无人工验证、学习与泛化证据不足 |

GPU阶段待办：共享模型实际SFT/DPO接入，完整回复/EOS条件序列求和及mask对照，reference冻结与梯度路径，BF16/FP32数值、batch/累积、optimizer/RNG恢复，模型训前效用门槛，匹配即时/普通future/历史future消融；随后才验证真实长上下文、吞吐和显存。当前新数据最长967 tokens，不替代上一轮20k历史的GPU验证。

CPU下一步优先增加不同源实例和真实交付轨迹、独立验证新增派生规范，并用键重命名/工具变化检验词汇捷径；不通过复制8条主样本来制造规模。

## 9. 交付索引

- [逐项完成审计](completion_audit.json)、[机器结果](results_summary.json)、[就绪性](readiness.json)。
- [训练主监督](training_v3/history_primary.jsonl)、[全部辅助数据](training_v3/all_preferences.jsonl)、[训练信号覆盖](SIGNAL_COVERAGE.md)。
- `dynamic_cce_v3/`、`finite_v2/`、`qwen_v2/`、`sequence_v3/`及对应审计文件为最终证据。
- `*_jobs*.json`和`*_receipts.jsonl`保存命令、时间上限、退出状态、源码哈希；全部原始版本保留。
- `FINAL_VALIDATION.json`、`final_manifest.json`、`changes.patch`保存交付核验及新增源码差异。
'''
(R/'REPORT.md').write_text(report,encoding='utf-8')
coverage='''# Training signal coverage v3

| Signal | Evidence | Quantity | Scope / limit |
|---|---|---:|---|
| Legacy ordinary reachability | Permission relaxation preserves strict direction | 3926 | One old shared workflow |
| Legacy ordinary compliant cost | Same intervention | 226 | Not historical flip evidence |
| Legacy CI-mediated future value | Direction disappears after permission relaxation | 24 | 12 reachability +12 cost; history causality unproven |
| New history-conditioned future value | One actual delivered-history intervention, both current actions lawful, direction reverses | 8 pairs /4 groups | Two source instances, two adapted workflows, one join grammar |
| New immediate CI | Valid alternative plus safe witness | 4 pairs | Canonical messages, synthetic norms |
| New other auxiliary future | Exact finite continuation | 196 pairs | Not all established as history-sensitive |
| Deprived-observation ambiguity | Same input, nonuniform strict preferences | 32 quarantined directions /10 classes | Never used as hard training labels |
| DPR | Executed counterfactual repair and success suffix | 158 states, 0 additional unique pairs | Only3 immediate-CI repairs; no independent DPR benefit |
| Non-exhaustive CCE | 192 comparisons, verified upper support where available | 74 preferences /118 UNKNOWN | Frozen finite reference controls, not arbitrary LLM support |
| Real model learning | No real LLM parameter updates | 0 | Frozen Qwen next-action diagnostics only |

Primary training file: training_v3/history_primary.jsonl. All208 rows remain
train-only at source-instance level. Finite-model horizon transfer is diagnostic,
not a new independent validation/test task split. No human norm verification.
'''
(R/'SIGNAL_COVERAGE.md').write_text(coverage,encoding='utf-8')
print(json.dumps({'tests':tests,'primary':8,'cce_preferences':certified,'cce_unknown':unknown,'real_llm_learning':False}))
