# KCI-Align 第二轮修复与CPU迭代报告

## Material Passport

本轮开始：2026-11-13 05:02:17 UTC。约90分钟授权窗口；每个实验worker/request上限不超过10秒，快检查不人为延长。研究目标仍是**共享执行模型在实际接收者知识条件下学习即时CI和长期合规任务策略**；CCE/DPR作为监督来源。原Proposal、Handoff、上一轮代码文件和产物保留，新模块与新版本证据单独保存。

**结论：本轮扩展了CPU实现和数据审计，但仍不应投入大批量LLM训练，方法有效性尚未建立。** 本轮增加了可审计的推断结构、重新认证的长历史监督、真实模型负结果与外源入口，也发现并修复了会污染监督的源语义缺陷。下述合成、有限模型、真实冻结LLM、公开来源锚点四类证据不能互相替代。

## 1. 本轮实际修复与核验

- **源语义一致性缺陷：** 新环境v1允许Fact结构value改变而可见text不变，能造成完全相同Prompt对应相反任务答案和偏好。v2使用唯一canonical renderer约束两者；独立34个注入反例均被拒绝。正常工厂样本的文本未变，全部112配置以v2重跑，v1结果及缺陷证据保留。
- **可观测标签gate：** 3552个安全可达非终局状态形成3152个精确Prompt类，其中352类有多个兼容状态。即时CI/菜单没有冲突，但48个同Prompt动作的未来可达性不同。96条严格偏好因此被隔离；没有通过多数票或任选隐藏状态产生标签。
- **序列接入：** 新 `kci/training_data.py` 将共享prompt、完整回复/EOS、padding、position IDs、分组和来源hash组成可复用CPU适配器；拒绝隐式截断、反向偏好、重复JSON键及metadata字段混入。不是HF Trainer或真实模型训练实现。
- **批量证据预算：** 新wrapper预声明比较和有限look，分配总体δ并在观察结果前持久化消费记录；失败不退款、重启不重复消费。它是可信调用者使用单一ledger的有限union-bound计账，不是anytime-valid推断、防篡改存储或自动权重冻结。

最终 **99/99 unittest通过**。fresh replay重新计算4860条raw偏好、3552个状态及1792条选定DPR诊断，4286条发布偏好的来源、方向、scope和观察gate均匹配。112个case worker加1个汇总worker全通过，最长2.425秒。详见[独立审查](research/REVIEW.md)、[最终重放](final_replay/REPORT.md)和[批量计账边界](batch_budget/REPORT.md)。

这里的48个可达性混叠并非隐藏接收者背景：A收到相同决策文本，却看不到R私有的task criterion，无法知道上游是否已经算错。正确分支上的“严格更好”在另一个兼容世界可能只是两个动作都无法挽回的tie。该问题在文本/结构修复后仍存在，必须隔离；它不是模型能力不足的证据。

## 2. 数据集：新增覆盖与仍然存在的独立性限制

| 项目 | 当前数量/状态 |
|---|---:|
| 固定主配置 | 112：两种答案、授权flip、数值变体、期限5/9 |
| 推断语法 | 4：同键关联、三前提alias chain、精确总量差分、完整候选集singleton交集 |
| 任务运算 | threshold、parity、membership |
| raw严格偏好 | 4860 |
| 观察不一致隔离 | 96 |
| 剩余同文本对重复 | 478 |
| 发布唯一偏好 | 4286 |
| CI / future来源 | 110 / 4176 |
| 兼容状态共同最优SFT控制 | 1982；另1170个观察类不导出 |
| 手写结构骨架 / 共同workflow / 生成祖先 | 4 / 1 / 1 |
| primary独立任务留出 | 无 |

数据：[preferences.jsonl](dataset_v2/preferences.jsonl)、[quarantine.jsonl](dataset_v2/quarantine.jsonl)、[SFT控制](dataset_v2/sft_controls.jsonl)、[数据卡](dataset_v2/DATA_CARD.md)。规则成立需要明确前提：差分要求同总体/时间/单位和精确整数，集合推断要求完整closed-world列表及唯一交集。空/多交集、异键、断链、错误scope不会被硬判为受保护关系；超出来源schema的情况拒绝。

四种结构共用一个角色流程。alias chain也可能只是join的多跳展开。因此不是四个真实业务领域，更不是112个独立任务。主评价仍无独立source/task-template留出；另做的4折leave-one-family-out只叫**共享workflow条件下的合成结构迁移**。参数、授权、答案、历史和数值变体跟随祖先组，不能随机按行拆分制造独立性。

每case仅取固定枚举顺序的前16个确认失败prefix做DPR，共1792次，不代表所有失败都已测试。1178个保留repair pair已在CI/future中出现（46个CI交集、1132个future交集），独立新增pair为0。因此此数据中的repair来源重采样不能证明DPR独立增益。所有动态规范均为手工合成合同，`human_verified=false`、`natural_language_semantics_verified=false`。

## 3. 长历史监督与截断预算

新增版本化的synthetic initial-ledger fixture，保留真实A SEND前缀并加入经固定公共语法验证的无关heartbeat。它们是声明的初始历史，不是256步Agent执行轨迹。所有长偏好重新运行CI、CCE及完整后果；没有把短样本旧标签直接贴到任意长文本上。

v1有98个唯一pair，但context/skip的事件数不同，不能排除长度捷径。最终matched v2让两侧都真实SEND相同数量context，仅改变一个关联键或scope；**80/80组的事件数、字符数及完整Qwen chat prompt token数完全相等**，同B消息的CI及严格偏好均反转。独立审计重编码全部160条侧面记录并独立推断通过。去重后140条，126条含较长历史，保留1–257条交付记录；完整pair最长21061 tokens。

这消除了这些配对中的纯长度差异，仍没有排除固定Record000/001、period_1/2等词汇捷径，也没有建立真实社会规范或实际LLM长上下文效果。数据与边界见[matched报告](environment/history_matched_v2/REPORT.md)。

| 完整序列预算 | 主数据保留 | matched长历史保留 |
|---|---:|---:|
| 512 | 1359/4286 | 8/140 |
| 1024 | 4286/4286 | 14/140 |
| 4096 | 4286/4286 | 56/140 |
| 8192 | 4286/4286 | 98/140 |
| 16384 | 4286/4286 | 98/140 |
| 32768 | 4286/4286 | 140/140 |

适配器对4286条主偏好、140条matched长偏好均完整编码通过。默认4k会拒绝matched数据的60%，不能通过删掉这些样本把长历史目标变成短上下文任务。32k只表示本例预处理能完整保留，不代表真实模型显存、速度、数值或行为通过。1824/4286/98/140是不同版本的处理量，存在共享来源与重复，不合计成独立训练规模。详见[序列接入报告](training_adapter/multi_v2_report.md)。

### 3.1 批次形状与模拟两rank负载

另固定72组算术模拟：3种负载、每rank batch 2/4/8/16、2种子、随机/全长度排序/随机大桶内排序。每对chosen/rejected始终同批，全部pair恰保留一次、零截断。120个短worker最长0.920秒；这不是实际DDP、显存或速度测试，也没有新训练。

下表为每rank batch=8时两种子的平均token-slot利用率（实际token数/补齐后槽位数）：

| 负载 | 随机 | 全长度排序 | 桶内排序 |
|---|---:|---:|---:|
| 主数据 | 77.53% | 99.02% | 97.55% |
| matched长历史 | 44.42% | 96.89% | 96.89% |
| 混合负载模拟 | 31.84% | 98.51% | 69.43% |

主数据两分支共4737660 tokens，matched数据共2339420 tokens；少量长pair也会占用大量槽位。混合仅是负载模拟，不合并为新的独立训练集。排序降低本例padding，却改变相邻family/监督来源的聚集：主数据B=8的batch内主导family份额由随机约52.75%升至全排序83.55%。matched B=16时，随机利用率约39.75%、两rank槽位相对skew约0.00365；全排序利用率约93.75%、skew却约0.325。故不能称排序同时保证负载平衡。matched B=8/16的桶大小超过样本数，桶排序实际退化为全排序。尾部不等量步骤另列，不掩盖或复制补齐；这些算术结果不证明真实DDP可直接使用该尾部调度。详见[批次形状报告](batch_shape_probe/REPORT.md)。

## 4. 有限共享模型训练：未得到方法优势

新代理是12744参数、24维共享tanh骨干与19类菜单动作的有限模型，输入只有可见Prompt的512维词/相邻词hash特征。没有手写CI闭包、风险标志或正确答案特征；仍不是LLM，也不生成自由文本。

4折×5种固定训练配置×2种子，共40模型。每折只用另外3个family的SFT/偏好；固定200步SFT、200步DPO（SFT控制除外）、batch16、β=1。没有用heldout结果选超参。独立核查训练批次与SFT均无heldout，跨split精确Prompt重叠0，同fold/seed的reference权重一致，U/E双梯度检查最大误差约3.72e-10。

| 配置 | heldout pair宏平均 | 初始执行成功 | 初始CI违规 | 合法前缀后成功 | 合法前缀后CI违规 |
|---|---:|---:|---:|---:|---:|
| SFT控制 | 82.0% | 80/224 | 0 | 120/448 | 112/448 |
| SFT→CI | 76.0% | 42/224 | 0 | 82/448 | 0/448 |
| SFT→future | 81.6% | 48/224 | 0 | 72/448 | 112/448 |
| SFT→全部唯一偏好 | 82.3% | 40/224 | 0 | 60/448 | 112/448 |
| SFT→来源重采样 | 81.4% | 60/224 | 0 | 90/448 | 112/448 |

每配置初始执行224回合（112参数化case×2种子）；这些不是224个独立任务。其A首动作全部为SKIP，避险本身可以是合理策略。零违规仅说明访问到的路径没有违规，不能证明其他合法历史下B仍安全。

因此用同一批冻结权重作后续覆盖诊断：先真实执行合法A SKIP或全context发送，再从B执行。每配置448回合中392可行、56在给定期限内已不可达。SFT/future/all/来源重采样各有112次违规，其中56发生在可行起点；CI-only无违规，却有310个可行任务失败和56个不可达任务失败。该干预不是新的独立测试，不用于重新调参。

再对80组等长相关/无关历史作冻结权重的下一步pair评分，全部40模型共1600个侧面读数正确率50%，800组joint flip为0，ties=0。**N=0已经失败**，不能将它归因于长历史独有退化；它说明该有限词特征模型没有学会这些关联键的条件翻转。该测试没有真实LLM、动作执行或新训练。详见[冻结历史诊断](text_history_stress/REPORT.md)。

这些结果没有显示“加入future/repair后稳定提升完整任务”的证据。来源重采样200批中CI/future/repair为67/67/66，只是改变已有pair的权重，不能当独立DPR收益。也不能据此否定更强模型上的KCI假设；当前容量、表示、共享workflow和短训练范围均有限。

## 5. 真实本地Qwen执行

沿用已安装Qwen2.5-7B-Instruct Q4_K_M，CPU实际size_vram=0，context4096。预声明16段（4family×yes/no×授权条件），全部初始状态有安全成功路径；固定seed/温度/菜单，无失败重试或事后调prompt。

| 结果 | 数量 |
|---|---:|
| 合规成功 | 2 |
| 确认CI违规 | 8 |
| 任务失败 | 2 |
| UNKNOWN | 4：3次非合约index、1次请求时限 |

44次task请求中43次完成，40个Action实际执行。已完成请求累计报告29236个prompt tokens、86个生成tokens；中断请求token成本未知。请求worker总245.48秒，最长8.715秒。10个确认失败均有离线DPR修正，但没有用于真实模型训练；因此不能声称模型已经学会修复。

模型pilot在v1工厂生成输入上运行；v2修复后逐字节核对44次prompt/menu、重放40个原Action和10个DPR结果全部一致，0次额外模型调用。模型驻留已释放，Ollama服务和WSL未改。新旧轮次任务不同，不能把4/12与2/16作受控性能下降比较；可以确认上一轮单模板零违规不支持对新增结构的外推。详见[真实模型报告](ollama_multi/REPORT.md)。

## 6. 外部来源锚点：已接入，尚未成为动态训练集

按原Proposal的数据路线，从固定版本官方来源保存了PiSAs Meeting/Jira各一套scenario/utility/appropriateness/visibility，以及MultiAgentPrivacy一个关系样本。两种PiSAs workflow共52个静态属性标签、16份初始资料；MultiAgentPrivacy样本有4agents、8tables、24rows。原始URL、revision、SHA及许可来源在[数据卡](source_anchors/DATA_CARD.md)。

三条anchor均保持 `delivered_events=[]`、`derived_ci_labels=[]`、`dynamic_train_ready=false`，不会被训练适配器接受，也未混入4286条偏好。源timeline只是初始资料；原子appropriate/sensitivity标签不能自动认证新增推断关系。

PiSAs固定版本的原生代码已只读核清：普通多代理、无private/shared memory模式下，默认preload按artifact.visible_to选取整份content并放入agent.context；属性visibility用于泄漏judge，不执行属性级输入过滤。两份scenario中11个属性存在artifact/attribute读者集合差异，涉及32个recipient–attribute组合；其中4个属性的值可在原始content逐字匹配，涉及14个组合。这确认了本例的输入暴露路径，尚未解决全部规范含义，也不推出整个benchmark无效。不能自行加属性ACL后称faithful复用。完整代码位置和固定commit见[原生暴露链澄清](source_anchors/ACL_CLARIFICATION.md)；来源数据卡保留早期待核验状态，后续结论以该澄清附录为准。MultiAgentPrivacy样本的high-income/significant-debt未给精确阈值，参考集合完整性也有具体歧义，不能直接作为精确task oracle。HF数据卡的许可声明与代码仓库许可分开记录；未找到许可声明不等于对研究读取作法律结论。

这些外源workflow是后续任务建设入口，尚无新增可执行动态任务、交付账本或经过独立验证的派生CI真值。没有因为保存了源文件就把动态任务独立性门槛改成通过。

### 6.1 外部任务效用的局部可执行核查

额外建立了只面向上述两个固定实例的约束校验：Meeting完整候选域3375个方案，基线约束保留6个、加入明确typed私有约束后保留1个；Jira完整候选域625个方案，技能/容量约束保留2个、加入cannot_be_sole_owner后保留1个。可行集合及唯一解均与固定源一致；独立审查另行重算全部4000方案并匹配。

候选生成与约束可行性判定不读源oracle；原始循环中的旁路canonical评分及解集事后对比使用oracle，不能称整个程序从未读答案。Jira容量来自私人artifact，因此基线可行数不表示Agent仅凭公开输入就能求解。删除任一关键私有约束均会重新产生2解；Jira的适当业务约束a22n未列入required_evidence且其sanitize文本未逐字出现在初始资料中，该列表不能独自充当可见充分证据。

这里通过的是受限canonical JSON任务合同。自由文本抽取、原始LLM judge、实际工具执行/交付/成本和由私有原因推出业务约束的语义仍未认证；新增动态偏好和CI标签均为0，两个实例仍dynamic_train_ready=false。详见[来源任务效用报告](source_task_probe/REPORT.md)。

## 7. 是否投入大批量实验

| 门槛 | 判断 |
|---|---|
| 指定有限环境、source一致性和回归 | 通过 |
| 已枚举兼容状态内的严格偏好可辨识性 | 隔离96条后通过；范围外/隐藏背景仍不保证 |
| 长序列完整保留与可复用CPU适配器 | 通过预处理合同；真实Trainer/模型未验证 |
| 有限批量统计预算接口 | 已实现；真实生成服务与审计校准未完成 |
| 两个固定来源实例的局部任务效用 | 约束可行性通过；尚无动态执行workflow或自然语言judge复现 |
| 批次padding与两rank负载 | 有限算术模拟完成；真实DDP/GPU未验证 |
| 独立、可执行、来源支撑的任务workflow | 未通过；当前训练仍1个共同workflow |
| 独立派生规范/自由文本语义验证 | 未通过 |
| 真实LLM训练与匹配消融增益 | 未进行，有限代理没有建立优势 |
| 真实长上下文行为、GPU数值/吞吐 | 未验证 |

**允许继续小规模来源接入、环境/标签审计和实现验证；大批量LLM训练仍NO-GO。** 下一步应先落实源workflow的实际暴露语义、执行工具和任务验证，独立审查派生关系规范，再采集可辨识的真实交付长历史。数据通过后，再在GPU上做小规模共享LLM SFT/DPO、即时/长期/repair匹配消融，以及token masking、数值、优化器、长上下文和吞吐验证。

## 8. 可复核交付与停止边界

- `results_summary.json`、`readiness.json`：机器可读结果和就绪性。
- `environment/`：v1/v2源快照、反例、四规则环境及两个长历史版本。
- `dataset_raw_v1/`、`dataset_raw_v2/`、`dataset_v2/`：历史原始结果、最终发布/隔离与分组。
- `final_replay/`、`research/`：重放、独立oracle、观察等价类、训练隔离、梯度、历史和前缀审计。
- `training_adapter/`、`batch_budget/`：序列接口与批量证据计账。
- `text_policy_v2/`、`text_history_stress/`、`text_prefix_stress/`：全部固定配置、权重、原始分数及执行轨迹。
- `ollama_multi/`、`source_anchors/`：真实冻结模型原始输出与公开来源字节、固定版本和边界。
- `source_task_probe/`、`batch_shape_probe/`：固定来源的局部任务校验和完整pair的批次负载模拟；[两轮因素覆盖表](scale_factor_coverage.md)汇总已测与未测范围。
- `input_snapshot.json`、`FINAL_VALIDATION.json`、`final_manifest.json`、`changes.patch`：旧资料保护、最后验证、文件哈希和新增代码差异。

早期源码语义缺陷、长历史测试错误假设、离线回放tuple/list序列化检查失败等均保留，不以重新运行后的成功覆盖负证据。所有新增实验均为本轮授权范围内的短worker；没有GPU训练、外部模型服务、全局依赖安装或未授权消息发送。

本轮停止在已确认的证据边界：继续重复同一有限模型或复制同一结构不能提供独立规范真值、真实执行任务或LLM学习收益。本报告保持原研究目标，将未完成门槛明确留给后续，不将工程通过包装成方法成功。
