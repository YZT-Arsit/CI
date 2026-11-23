# KCI-Align：v3.1 数据接入、语义动作表示与单一共享 4B 策略验收

日期：2026-11-23（4090 服务器日期）。本报告为开发过程审计，不与论文独立盲测结果混报。

## 研究决定摘要

- **数据不是笼统不足。** v3.1 有 21 个业务×知识图根、63 个配置、315 个模型可见早期状态、126 个严格平衡反事实状态；训练 72 个早期长期偏好，另外 485 个训练图多角色原生示范。它主要监督 Coordinator 的 OPEN/SKIP，而旧首次披露实验直接监督 SourceAgent 的 SEND_RAW_R/VERIFY_PRIVATE_R；保留两种来源与规范合同。
- **动作接口已经修复。** 从选项字母分类改为语义动作逐项评分。每个候选使用相同的“候选动作—A接受/B不接受”固定二元答案，并只用同一个 LoRA 计算分数；原生候选按照语义 ID 排序。任意原始菜单排列得到逐候选逐字节相同的 prompt，消除“选项位置语义映射”歧义。
- **共享架构已经真实实现。** Qwen3-4B-Instruct-2507 的一个冻结底座加**一个共享的 trainable LoRA**。角色由当前授权 `env.observation()` 给定，不加载或切换角色专属 adapter。单模型已经执行六类角色的 142 次决策及 316 次原生转移。
- **算法区分仍须保留。** 本轮 CE 是正确接口的训练可行性/强基线，并非论文全部 CCE+DPR+DPO；历史奖励和旧 Qpi/原生 Qstar 的证据等级不能混合。

## 1. 数据合同复查

V3.1 在 `artifacts/20261120_v31/V31_GATEA_FINAL_CANDIDATE`。新模型输入由规范和原生观察转换，未注入 Qstar、future outcome 或隐藏违规标签。

| 数据 | 模型状态数 | 独立业务图根 | 角色/监督 | 训练用途 |
|---|---:|---:|---|---|
| V31_TRAIN_72 | 72 | 12（4图家族） | Coordinator preflight，精确原生 Qstar OPEN/SKIP | 允许 |
| V31_EVAL_54 | 54 | 9（开发1+探索2图家族） | 同一预行动，精确原生 Qstar 仅评价 | 禁止训练 |
| RELEASE_TRAIN_STRICT8 | 8 | 2 | 来源角色第一次披露，原生 CI + 固定安全续行，4 RAW / 4 VERIFY | 允许 |
| RELEASE_EVAL_MATCHED12 | 12 | 3 | DG14 / DG18 / DG19，分根评价 | 禁止训练 |
| MULTIROLE_TRAIN | 485 | 12 | 五类 Agent 的原生最优动作示范，不是模型 Qpi | 允许作为独立来源 |

新接口重新审计 v3.1 的 63 个完整历史对，**63/63** 的非历史提示内容和合法候选一致，只有已送达收据内容不同；**63/63** 形成 OPEN vs SKIP 排序翻转。训练与评价按知识图家族分组，已见探索测试不作盲测声明。文件：`V31_STRICT_RENDERER_VALIDATION.json`。

## 2. 统一动作编码

当前实现位于 `tools/kci_shared_semantic_action.py`。

定义 `eθ(o,a)=log pθ(A|o,a)−log pθ(B|o,a)`，`πθ(a|o)=softmax_{a∈A(o)} eθ(o,a)`。这里的 A/B 是**对每个语义候选固定含义的二元评分标签**，不是原来用户行动菜单的动态选项位置。完整合法动作集合排序后统一显示；同一可见状态每个动作单独评分。

DPO 的相同共享表征及冻结参考模型接口见 `tools/kci_shared_semantic_dpo.py`，其偏好 margin 为 `[eθ(o,a+)−eθ(o,a−)]−[e0(o,a+)−e0(o,a−)]`。它兼容论文 §4 中一套共享策略的同状态偏好目标，但 D_repair 仍须从实际失败轨迹经验证后额外构建，不能拿 485 条脚本示范直接冒充。

语义动作选择使一次决策需要为每个候选运行一次语言模型前向；这项推理成本要在最终论文报告中计入，不能与先前一次字母预测等价报时。

## 3. 最小接口拟合：真正训练后的动作变化

模型：Qwen3-4B-Instruct-2507，NF4，BF16，LoRA r=8/alpha=16，seed7，学习率 2e−5。旧实验修正前的平衡 CE 在首次披露严格 matched 训练状态为 4/8、整对 0/4。

新语义动作 CE 使用相同 8 个严格 matched 训练状态、96 次 optimizer 更新、batch2。训练动作正确 **8/8**，真正整对双向正确 **4/4**（2独立训练图根），换位一致性由 prompt 构造完全保证。

该适配器在来源任务开发 DG14 为 **2/4**，已见探索 DG18/DG19 为 **2/8**；DG18 的严格历史对 **0/2** 完全正确。该结果支持此前的动作位置表示是训练障碍，同时说明单靠两根监督不具备已证实的跨图迁移能力。原始结果和权重：`train_shared_release8_seed7`、`eval_shared_release8_seed7`。

## 4. 多来源共享模型：与角色 LoRA 路由完全区分

新的 joint CE 由**同一个新初始化 4B LoRA**训练，144 步、batch3，三个训练来源各抽样144次。V31 使用全部12训练业务根；多角色示范经过根×角色轮转，实际覆盖12个训练根和 Coordinator、DirectoryAgent、OrderAgent、RecordsAgent、ComplianceAgent 五种角色。保存的训练明细在 `train_shared_joint3_seed7/TRAINING.json`。

| 状态动作指标 | 新共享 joint CE |
|---|---:|
| 首次披露 train | 6/8 |
| 首次披露 train 整对 | 2/4 |
| 首次披露 DEV | 0/4 |
| 首次披露已见探索 | 2/8 |
| V31 preflight train | 42/72（整对8/36） |
| V31 preflight DEV | 11/18（整对2/9） |
| V31 preflight 已见探索 | 21/36（整对3/18） |
| 多角色训练来源原生状态动作（仅实际抽样144次参与优化） | 392/485 |

多角色状态准确率主要衡量原生动作示范的模仿，不能替代动态历史翻转；跨任务 CE 目前明显干扰首次披露的小样本拟合。此处应诊断来源任务权重、反馈证据品质与同状态偏好目标，不能因新增全部485条示范就推断 CI 会提高。

## 5. 一个共享模型的原生团队轨迹

`tools/kci_shared_semantic_native_rollout.py` 在实际环境里给所有需要模型行动的角色调用同一个 LoRA。统一的可认证源读取、业务路由和单步工具动作仍按共同固定执行器运行；状态变化、接收账本与原生 CI 判决来自真正环境。

| 划分 | 原生合规任务完成 | 原生 CI 违规 | 业务×图根 |
|---|---:|---:|---:|
| TRAIN | 14/16 | 2/16 | 4 |
| DEV | 4/4 | 0/4 | 1 |
| 已见探索 | 8/8 | 0/8 | 2 |
| 合计 | **26/28** | **2/28** | **7** |

真实记录总计**28条完整轨迹、316次状态转移、142次单共享策略的模型决策**，决策涉及 RecordsAgent32、Coordinator28、DirectoryAgent28、BusinessExecutor26、OrderAgent20、ComplianceAgent8。个别训练根发生直接实际 CI 违规，因此 joint CE 不能宣称保障 CI，也不能拿开发/已见探索表现代替盲测泛化。

## 6. 本轮主线研究决定与边界

1. 接受**语义动作评分替代位置字母选择**作为此项目新共享执行接口；旧的按角色 LoRA 仅作诊断历史，不进入最终架构。
2. 接受 v3.1 的63组真实历史双向对作为本轮更大覆盖的开发数据；独立业务根与仅7类规则图须分别统计。
3. 由于 joint CE 已见明显任务间干扰，**下一主实验应采用同状态、证据过滤的共享 DPO/CCE 偏好**，并把实际失败点 DPR 与历史状态监管分别作为数据来源，禁止混合原生最优、Qpi采样与示范而不标注。
4. 新根盲测仍需独立封存；现有探索根都参与过开发审计。现阶段不讨论更大模型容量。
5. 强显式规则闭包规划器仍是当前合成语言的必要基线；最终论文须将原生规则已知与学习模型实际推断区分。

## 7. 复现证据

仓库根：`D:\research\longhistory`。本轮数据和训练产物根：`artifacts/20261123_shared_semantic_action_v1/`。4090 原始目录：`/root/kci_native_4090_20261122/artifacts/20261123_shared_semantic_action_v1/`。

核心脚本：`tools/kci_shared_semantic_action.py`、`kci_shared_semantic_data.py`、`kci_shared_v31_history_audit.py`、`kci_shared_semantic_train.py`、`kci_shared_semantic_eval.py`、`kci_shared_semantic_native_rollout.py`、`kci_shared_semantic_dpo.py`。原始冻结输入有 SHA256 manifest，`train_*/TRAINING.json` 保存模型版本、优化配置和实际采样范围；`eval_*/DECISIONS.jsonl` 为每个原生语义动作及概率；`native_*/WORLD_EPISODES.jsonl` 保存逐步真实判定和状态哈希。根和结构图分层统计见 `eval_*/PER_ROOT.json`。

全量本机测试：`cd /d D:\research\longhistory\code && set PYTHONPATH=D:\research\longhistory\code;D:\research\longhistory\tools && python -B -m pytest -q tests`，此前验收 **403 tests、395 subtests**，新增共享语义 DPO 单元测试另有3条。

## 8. 同一共享动作表示上的 DPO 检验（追加，真实 4090）

在上述单一共享 LoRA 架构和语义动作评分下，执行了进一步的**共享参数、冻结底座参考策略 DPO**：`tools/kci_shared_semantic_dpo.py`，96 个真实 GPU 更新、batch2、seed7、LR 2e-5、beta2。训练数据只用 v3.1 train 的 72 个原生 Q-star 同状态早期 OPEN/SKIP 监督和首次披露 train 的 8 个 CI+固定安全续行监督；没有从开发/探索标签中训练，也没有按角色另加载 adapter。它是论文 §4 偏好更新算子的**部分方法原型**，还不是包含真实 on-policy CCE/DPR 的完整最终算法。

DPO 的训练损失在首步 0.693 后未稳定下降，96 步时为 2.00876，末步更新前梯度约 91.19（实际进行 grad-norm clip=1）；偏好正确批内数量在后期为0/2。冻结后，所有首次披露状态选择 VERIFY、所有 v3.1 preflight 选择 OPEN，说明走向保守的固定动作捷径：

| 监督/评价区域 | 共享多来源 CE 动作正确 | 共享原生偏好 DPO 动作正确 | DPO 双侧同时正确 |
|---|---:|---:|---:|
| 首次披露 train | 6/8 | 4/8 | **0/4** |
| 首次披露 dev | 0/4 | 0/4 | **0/2**（此组无应变历史翻转） |
| 首次披露已见探索 | 2/8 | 2/8 | **0/4** |
| v3.1 train | 42/72 | 36/72 | **0/36** |
| v3.1 dev | 11/18 | 9/18 | **0/9** |
| v3.1 已见探索 | 21/36 | 18/36 | **0/18** |

注意：首次披露 dev DG14 两种历史本来都合法，`0/2` 只是双方均选择更优低成本动作的成对准确数，不是历史 flip 失败数；真正要求翻转的 strict DG18 两组在 DPO 下 **0/2**。对 v3.1 所有历史对，零双侧正确是决策退化，不是数据不存在双向教师。

在一个共享 4B LoRA 控制全部角色的原生完整链上，DPO 28/28 合规完成、0/28 违规（train16/16、dev4/4、已见探索8/8），共 348 次真实原生状态转移和144个模型决策。相较 CE 的 26/28 合规完成、2/28 违规，DPO 避开了原文泄漏，但其强制核验多付出执行开销：train平均步数 15.25 vs CE 13.00，dev 16.00 vs14.00，探索17.00 vs16.00。**只看合规完成会误判 DPO 成功。** 它没有学会有区别地合法披露，也没有达到 v3.1 长期决策历史翻转目标。

**研究决定（冻结）：** 接受已验证的共享语义动作接口和单共享 LoRA 执行架构；v3.1 可以作为有效的控制数据。拒绝将当前等成本/离线原生偏好 DPO 作为论文效果。下一开发检验应固定真实历史状态匹配的偏好证据和实执行政策续行 Qpi，逐角色记录有效/弃权、成本与安全完成，并仅从真实失败轨迹生成 DPR；另外明确平衡合法披露与防止无依据过度核验。先完善目标/监督，不把失败归因于4B容量或简单增加 prompt改写。

原始 DPO 文件：`artifacts/20261123_shared_semantic_action_v1/train_shared_DPO_CI_future_seed7/TRAINING.json`、`train_shared_DPO_CI_future_seed7/adapter/`、`TRAIN_SHARED_DPO_CI_FUTURE.stdout.log`、`eval_shared_DPO_CI_future_seed7/{DECISIONS.jsonl,SUMMARY.json,PER_ROOT.json}`、`native_shared_DPO_CI_future_seed7/{WORLD_EPISODES.jsonl,SUMMARY.json}`、`EVAL_NATIVE_DPO_CI_FUTURE.stdout.log`。

最终全量本机回归：**406 tests + 395 subtests 通过**，共享模型的语义位置不变性、原生多角色路由和 DPO 数学合同均有测试。完整运行元数据、原始判断和未见图开发标记均保留，不以开发/已见探索结果声称独立盲测泛化。
