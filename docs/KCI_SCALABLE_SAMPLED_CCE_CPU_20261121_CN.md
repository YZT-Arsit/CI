# KCI-Align｜面向大模型放大的小资源算法研发（2026-11-21）

## 一、这次真正解决了哪一个实施瓶颈

原始 CI Insight 保持不变：接收者 R 的已送达知识改变后续行动的合法性、长期任务价值与下游责任位置。

2026-11-21 前一轮已经证明：
- 在动态 A→R→B→C 的真实原生执行中，原小型共享策略在 A 的严格长期最优行动上只有 2/6；
- 参考最优 Qstar 和当前执行团队的实际 Qpi 可给同一合法上游行动反向价值排序；
- 直接把 Qpi 偏好混入同一个共享神经网络，会把下游已学会的安全决策改坏；
- 使用角色隔离和先下游、后上游的训练可以让三个随机种子在已见探索图上实现 8/8 安全完成、6/6 严格最优；
- 但此前**训练期间调用精确原生 Qpi 穷举**，这一点不能直接放大到 4B LLM。

**本轮解决的核心技术问题**：不用求出精确 Qpi，也不用向模型输入隐藏 Qstar，能不能只用少量真实多 Agent 续行生成可信偏好并完成实际参数训练？

答案是在本受控 CPU 环境中**可以**，并且进一步完成了一套不需全局最优专家动作标签的三阶段偏好训练原型。

## 二、真实策略续行采样：对照精确 Qpi 的误判率

冻结已有 3 个种子的原始 SFT-CF 共享 actor 或下游残差修正后的 actor。同一个上游 A 状态分别强制两个当前合法的 source package，然后完全交给该冻结 actor 使用原生工具和后续 role actions。两分叉各采样 K 条完整终局轨迹。

比较：
- K = 2/4/8/16；
- common-random-numbers（同一续行随机种子）与独立随机种子；
- 3 个模型 seed，3 个采样重复，原生 16 train /4 dev/8 developer-seen test。
- 预定过滤：|mean return gap|>=0.01、配对方向一致比例>=0.75。它是启发式信号过滤，不是统计上证明 95% 正确的置信区间。
- 精确 Qpi 只在采样完成后独立计算，用于审计偏好方向；采样与训练本身不访问该精确值。

**训练图结果（3 model seeds × 3 sample replicates；注意每种子内场景相关且有包名交换）：**

| 冻结的下游策略 | 每动作 K | 接受的偏好数 | 接受中相对精确 Qpi 方向错误 |
|---|---:|---:|---:|
| 原 SFT-CF | 2 | 98/144 | 25 |
| 原 SFT-CF | 16 | 84/144 | 12 |
| 下游 residual 修正 | 2 | **126/144** | **0** |
| 下游 residual 修正 | 16 | 125/144 | 0 |

这里的重要机制不在于 CRN 技巧。下游修正后，独立随机采样与配对随机采样表现几乎一致；在下游未修正前，两种采样都有明显误判。**下游真实执行能力的提高，显著降低了有限预算下反事实教师的噪声**。

全采样审计累计 32,256 次实际 native 终局续行和 288,721 个 native state-action steps；验证了 policy 冻结及无 oracle 混入训练教师的接口。

## 三、将采样偏好真正送入上游 DPO

保持同一个下游残差修正模型、同一训练世界和 80 次 A-DPO 更新。训练偏好仅由被冻结的真实策略采样给出。

对 K=2/8/16、独立或 CRN 续行，3 seeds × 2 sampling repeats，共 **36 次实际 CPU 训练**：

- 36/36 个训练结果在探索测试均 **A 严格最优 6/6**；
- 36/36 均 **完整贪心轨迹 8/8 合规完成**；
- 三 seed 平均完整随机策略 expected return 约 **0.6454**；
- 单次 K=2 训练实际接受 14 条严格训练偏好；2 条价值基本平局不强行输出方向；
- 精确 Qpi 全树监督基线此前结果约 0.64538，二者在这套合成数据中几乎一致。

**资源换算**：每个 A 状态两个行动各 K=2，只需 **4 条完整冻结策略续行**；固定 K=16 需 **32 条**，前者全轨迹调用数为后者的 1/8。真实 4B 环境中的 token 成本仍取决于各续行长度，不能把这个比例直接称为 GPU 时间节省。

K=2 paired 的三个可复现模型已经单独保存权重，从磁盘重新加载并重新执行完整 native 环境，确认 6/6 A、8/8 全链合规和 exact expected return 可重算。权重哈希与来源样本哈希在单独审计文件中。

## 四、自适应预算的可扩展性检查

为了避免未来 LLM 上简单固定 K=16 的过高开销，实现了一个小型逐级预算门：

- 从 K=2 开始，对高幅度且完全同向的证据直接接受；
- 不足以判断时追加到 K=4、8、16；
- 最大 K=16，不足则不产生偏好。

注意该门是**工程探索性门限**，不是独立预注册的算法优越性证明。

在 3 seeds ×3 sample replicates 的 9 次上游 DPO 训练中：
- 每个训练世界平均 K=5.56；
- 每个 seed 的 16 个训练世界平均 177.8 条额外完整续行，固定 K=16 需512条，约省 **65.3%** 的续行调用；
- 总共未出现被接受的训练偏好方向错误；
- 9/9 训练的模型均在探索测试上 6/6 严格最优、8/8 合规完成，平均随机策略回报仍接近0.6454。

当前数据下固定 K=2 已足够且更节省；自适应采样的价值在更随机、更难判别的未来 LLM 状态中仍需再测，**不应声称它已在本任务优于 K=2**。

## 五、进一步完成“不使用全局 Qstar 最优专家动作”的三阶段完整训练

为了避免最后一条可放大的障碍（下游依然由全局最优参考 SFT 提供答案），新建了另一条训练链：

1. 从原生训练状态中实际执行验证得到 **28 条同状态 RAW 违规 / VERIFY 合法修正对**，只用这些即时违规与合法纠正证据训练来源 Agent 的局部 DPO 残差。它是**可执行状态分叉下的局部纠正**，不是自然采样失败轨迹上的完整 DPR。
2. 冻结来源 Agent；B/Coordinator 的当前合法 OPEN/SKIP 分叉**每个动作只进行 K=4 次完整 Qpi 续行**，经过方向过滤后产生 B-DPO 偏好；每 seed 查询32个候选状态，**32/32有有效偏好**。
3. 冻结 B 和来源 Agent；A/RecordsAgent 的两个合法 evidence-package 分叉同样 K=4，16个候选状态里 **11/16有有效偏好**；对独立 A 头训练 DPO。

三个 model seeds 下完整随机策略的平均期望回报：

| 训练阶段 | Seed 7 | Seed 1234 | Seed 2025 | 三种子平均 |
|---|---:|---:|---:|---:|
| 冻结原 SFT-CF | 0.54319 | 0.53653 | 0.52865 | 0.53613 |
| 只做局部 release 纠正 | 0.53409 | 0.53136 | 0.53081 | 0.53209 |
| 追加 B 的有限 Qpi 续行 DPO | 0.59350 | 0.59235 | 0.59126 | 0.59237 |
| 追加 A 的有限 Qpi 续行 DPO | **0.63677** | **0.63623** | **0.63615** | **0.63638** |

最终所有三 seed：
- 贪心完整业务成功且无 CI 违规 8/8；
- A 严格长期最优 6/6；
- 平均 native 软策略 CI 违规概率从原 SFT-CF 的 **5.37%** 降到 **约0.507%**；
- 完整 soft-policy expected return 较初始提升 **约0.1003**；
- 每 seed 的 B/A 反事实采样总预算为 32×2×4 +16×2×4 = **384 条完整多 Agent 续行**，额外本地 raw/verify 违规验证约 28 对。
- 全程不需要知道训练状态的理想 Qstar 最优行动，也不需要穷举当前 Qpi 作为偏好教师。

值得保留的负结果：只训练 release 纠正，平均回报并未提升；仅修改下游一处安全行为不能等同长期协作训练，需要后续 B/A 的真正信用分配。

为了保证角色边界，训练使用同一冻结共享主干，R、B、A 的三个**相互隔离可学习的小头**。这是将来一个基础 LLM + 分角色 LoRA/adapter 的结构原型，而不是把结构化 Horn 小头当成最终 LLM。

## 六、独立复验

1. 新增的核心续行测试与历史 CI 修复回归合计 **14 passed**。
2. 有限 K=2 样本 DPO 的 3 个保存模型重载及原生 replay PASS。
3. 最终三阶段模型 3 个保存 checkpoint 独立加载，**36 条非训练完整 native 轨迹**、**426 个实际工具与 Agent 动作**及36个完整 soft-policy expected return 重新计算 PASS；A 更新不改变下游评分的角色隔离在独立代码中检查；28个同状态即时违规纠正对重新生成并验证。
4. 完整分割 16 train /4 dev /8 developer-seen test；测试图此前已被开发人员查看，不能当独立盲测。
5. 使用已知 synthetic Horn 规则及角色可见事实，不能把它描述为大模型已经学会开放域自然语言 CI 推理。

## 七、可放大到 LLM 的明确算法框架

优先实施**role-conditioned / capability-staged counterfactual DPO**，而不是恢复旧的全序列终局 GRPO+Qstar 奖励加权：

- **Shared pretrained backbone**：冻结或 QLoRA；以角色标签路由轻量 adapter，至少将 source-release 修复、Coordinator preflight 和 RecordsAgent A 的训练信号隔离。CPU 已直接证明全共享参数混训可让 OOD 的OPEN/VERIFY变成SKIP/RAW造成CI泄漏。
- **Local causal repair**：只收集本来被执行/可重放的下游实际隐私违规状态，构造同状态、同角色、可执行的合法修正，不惩罚先前合法上游披露。真实 LLM 阶段优先使用 on-policy failed rollouts + restart。
- **B-level CCE**：冻结已经局部纠正的来源角色，比较两个合法 B 前缀在相同后续冻结团队的真实奖励；先用小 K、再按证据追加续行。DPO 更新只打到 B role adapter。
- **A-level CCE**：在 B/来源技能稳定后，同样比较两个合法上游披露并训练 A adapter，不把当前弱下游策略当作永久教师。
- **Refresh**：仅当更新下游执行策略后重新采集依赖它的上游信用；单个 preference collection epoch 内冻结参考策略和记录模型/状态哈希。
- **Cost accounting**：训练时记录分叉个数、实际每分叉 rollouts、生成 token 与 tool actions、有效 pair 数、过滤比例，以及每个角色的 action log-odds 增幅。

当前 CPU 证据只说明**方法实现和机制可行**。面向 ICML 的下一步是把这个无精确 Qpi / 无理想 Qstar 训练教师的流程，在相同底座的可微调 LLM 上跑起来，并和普通 SFT、全共享直接 Qpi、终局 RL、理想 Qstar 教师做相同环境对照。不能将这些已见 synthetic 图的 8/8 直接称作独立泛化。

## 八、代码与原始数据目录

- tools/run_kci_budgeted_counterfactual_cpu.py：冻结策略、真实原生续行、K/独立随机或CRN对照；
- tools/run_kci_budgeted_mc_dpo_train_cpu.py：采样的偏好真正训练A并评价；
- tools/run_kci_adaptive_mc_cee_cpu.py：逐级使用更大采样预算；
- tools/audit_kci_sampled_budget_dpo_cpu.py：保存 K2 决策头、重新加载并回放；
- tools/run_kci_native_repair_mc_bridge_cpu.py：28 下游违规纠正 + B/A 各 K4 的完整三阶段；
- tools/audit_kci_native_repair_mc_bridge_cpu.py：最终 checkpoint 独立重放；
- code/tests/test_kci_budgeted_counterfactual_cpu.py：采样函数、采样可复现性和状态合法性回归。

关键原始结果：
- artifacts/20261121_kci_budgeted_counterfactual_cpu_v2/RESULTS.json
- artifacts/20261121_kci_budgeted_training_cpu_v1/RESULTS.json
- artifacts/20261121_kci_adaptive_sampled_cee_cpu_v1/RESULTS.json
- artifacts/20261121_kci_budgeted_dpo_saved_audit_v1/AUDIT.json
- artifacts/20261121_kci_native_repair_mc_bridge_cpu_v1/RESULTS.json
- artifacts/20261121_kci_native_repair_mc_bridge_cpu_v1/INDEPENDENT_SAVED_MODEL_AUDIT.json

训练、数据、角色冻结及完整回报都已落到具体可重跑的代码和记录中。所有新结果与此前的 V3.1 原始数据独立保存，未修改 main 分支。
