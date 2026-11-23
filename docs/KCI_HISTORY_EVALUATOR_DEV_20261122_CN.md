# KCI 历史可见评价器 vs 去历史评价器：开发实验封口记录

## 1. 本轮唯一实验格及决策门槛

**格子**：在原生 SourceAgent 第一次向 reviewer R 披露前的决策状态中，比较“评价器可见真实已送达接收者历史”与“评价器不可见该历史”。

**唯一的评价器变量**：是否向离线标签生成器提供 R 已成功送达的事件记录。两臂使用相同的世界、真实 SourceAgent 观察、任务目标、规范、候选行动、原生任务验证器和后续固定安全续行。历史屏蔽只在评价器克隆状态中完成，真实角色输入始终包含真实历史。没有把秘密标签或原生 oracle 值注入角色提示。

**事先冻结的分流**：① 若标签不能产生同一原文在不同历史下的合法/违规双向监督，停止在数据/奖励层；② 若策略能拟合训练根而新根不迁移，下一阶段查覆盖；③ 若清晰的训练对仍学不动，优先查动作表示，再讨论模型容量。本轮不扩大算法或训练数据。

## 2. 直接阻碍本格有效性的修复

- 修复 `tools/kci_llm_text_surface.py`：此前把 `entails=X/Y/Z` 的中间推断也渲染成“推出受保护关联”，影响 5/7 个原始图根。现在把中间事实明确称为 intermediate fact，只有 `entails=protected association` 才写 restricted association。原生规范、知识闭包及 verifier 均未修改。
- 冻结完整角色输入与标签哈希；对 DPO 和动作交叉熵训练都使用相同的原生动作集、同一 Qwen3-4B-Instruct-2507、NF4/BF16、LoRA rank 8/alpha 16、seed 7、64 步、batch 2、学习率 2e-5。
- 针对已发现的选项位置捷径，每一个训练样本均提供 A/B 原位与交换位两种动作菜单；推理统计同时保留原位、交换位和按语义动作平均的决策。
- 回归测试：`python -B -m pytest -q code/tests/test_kci_llm_view_ci_gate.py code/tests/test_kci_history_evaluator_cell.py`，**7/7 通过**；`git diff --check` 通过。

## 3. 原生评价器：分子、分母与图根

这里的“世界”是同一源图根经 source mode 和选项命名交换得到的派生世界。不能把派生世界数当成独立图根数。

| 分层 | 成对派生世界 | 历史可见教师：双向有效配对 | 去历史教师：正确双向配对 | 提供双向配对的独立根 |
|---|---:|---:|---:|---:|
| 全部原生状态 | 28 | **12/28** | **0/28** | **3/7**，DG01、DG05、DG18 |
| **严格 matched**（去除送达行后其余提示逐行相同） | **14** | **6/14** | **0/14** | **3/7** |
| matched · train | 8 | 4/8 | 0/8 | 2/4：DG01、DG05 |
| matched · dev | 2 | 0/2 | 0/2 | 0/1：DG14 |
| matched · exploratory test（开发者此前已查看） | 4 | 2/4 | 0/4 | 1/2：DG18 |

全部 56 个单侧状态中，历史可见教师标签为 `VERIFY_PRIVATE_R` 12、`SEND_RAW_R` 44；去历史教师为 `SEND_RAW_R` 56。严格 matched 的 6 对中，风险历史发送原文 **6/6 原生违规**，安全历史发送同一原文 **6/6 合规且任务可完成**，在安全侧发送原文均比私有核验少 **1 个原生步骤（6/6）**，私有核验在两侧均存在安全完成续行。

**关键审计限制**：另外 6 组两向翻转处于 `tradeoff`，安全与风险状态除已送达收据外还存在 `Remaining work budget` 差异；它们可以计入教师可识别标签差异，但**不能**计入“只改变接收者历史”的严格反事实证据。完整逐行差异在 `NONHISTORY_MATCH_AUDIT.json`。

## 4. 模型是否真正按历史行动：三次匹配诊断

**训练标签**：原始 train 有 32 个状态、16 个派生世界、4 个独立根；历史可见教师为 8 VERIFY / 24 RAW，去历史教师 0 VERIFY / 32 RAW。再从 DG01、DG05 提取 16 个风险/安全双侧状态（8 VERIFY / 8 RAW，对应去历史 16 RAW），以排除原始 1:3 类别不平衡。严格纯历史 matched 的训练子集另冻结为 8 个状态、4 个成对派生世界、2 个根（4 VERIFY / 4 RAW；去历史 8 RAW），**未单独训练该 8 状态子集**，留作下一次动作表示修复后的确定性回归输入。

| 冻结诊断 | 历史可见臂：32 个 train 状态正确 | 去历史臂：32 个 train 状态相对于原生真值正确 | 严格 matched train 双向全部正确 | 严格 DG18 matched 双向全部正确 |
|---|---:|---:|---:|---:|
| DPO，原始 32 状态训练 | 8/32 | 8/32 | **0/4** | **0/2** |
| 直接动作 CE，原始 32 状态训练 | 24/32 | 24/32 | **0/4** | **0/2** |
| 直接动作 CE，平衡 16 状态训练 | 12/32（含未用于训练的 16 状态） | 24/32（同一套冻结状态） | **0/4** | **0/2** |

前两种配方分别退化为“始终 VERIFY”和“始终 RAW”。平衡 16 状态训练后，历史可见臂在训练源图的全部 8 个双向配对里只有 2/8 出现语义动作变化，但 **0/8 同时正确**；其中完全 matched 的四组 **0/4**。对纯历史 matched 的 8 个状态，历史可见臂仅 **4/8** 动作正确（违规侧 2/4、合法侧 2/4），交换选项位置会使 **6/8** 的语义动作变化；在原始 32 个训练状态中，该敏感性为 **28/32**。

**因此没有证据说训练已经拟合，不能把 DG18 的 0/2 解读为“训练已成功但新根不迁移”。** 更不能用训练损失趋近零替代真正的动作测试。此处是固定第一个披露决策 + 真实 verifier 和固定安全续行，**不是完整 4B 多 Agent 自由 rollout，也不是 ICML 主表结果**。DG18 在本轮之前已被开发者查看，不是盲测新根。

## 5. 本轮研究决定

1. **评价器门槛通过**：可在原生可检查规范下导出严格纯历史的双向、任务有用监督，口径是 **6/14 matched 派生世界，3/7 独立根**。这不是 6 个独立训练任务。
2. **当前 4B 动作学习门槛未通过**：DPO、普通动作 CE 和平衡 CE 都未在训练根形成一个完全正确的历史双向配对（严格 0/4；平衡全部 0/8），因此不继续扩大旧配方，也不报告模型泛化提升。
3. **下一步优先查动作接口与位置不变性**：28/32 的换位语义改变是直接证据。用已冻结的 8 状态纯历史对进行动作表示/输出映射的最小回归，不先扩大数据或比较更大参数量。只有能拟合训练对再考查 DG18 的跨根迁移；若固定动作表示后仍不能拟合，才检查模型容量。A 角色既有 6 个成本捷径偏好未在此格中被声称修复，列为后续独立工作。

## 6. 原始证据与复核入口

Windows 工作树：`D:\research\longhistory`

完整原始记录目录：`D:\research\longhistory\artifacts\20261122_history_evaluator_dev_v1\`

- 原生评价器 56 状态、两侧候选的原始续行和验证哈希：`PAIR_STATES.jsonl`；汇总：`SUMMARY.json`。
- 纯历史反事实检验：`NONHISTORY_MATCH_AUDIT.json`；全部 28 世界包含关系与匹配判定。
- 固定两臂训练记录：`TRAIN_VISIBLE.jsonl`、`TRAIN_BLIND.jsonl`、`DATA_MANIFEST.json`；平衡16：`TRAIN_*_STRICT16.jsonl`、`STRICT16_MANIFEST.json`；严格纯历史8：`TRAIN_*_STRICT8.jsonl`、`STRICT8_MANIFEST.json`。
- 三次模型评价的逐状态两种位置概率、贪心动作、原生后果、原始标签：`gpu_eval_dpo_s7\DECISIONS.jsonl`、`gpu_eval_ce_s7\DECISIONS.jsonl`、`gpu_eval_strict16_ce_s7\DECISIONS.jsonl`；各目录另有 `SUMMARY.json`。
- 根分层分子/分母及三种模型的纯历史 matched 结果：`FINAL_DEV_CELL_SUMMARY.json`；DPO/CE 对应的 `MARGIN_AUDIT.json`。
- 两臂 GPU 原生训练日志：`TRAIN_BOTH_RETRY.stdout.log`、`TRAIN_BOTH_CE.stdout.log`、`TRAIN_BOTH_STRICT16_CE.stdout.log`；对应 `EVAL_*.stdout.log`。
- 训练模型 LoRA adapter、`TRAINING.json` 位于对应的 `train_*\adapter\` 目录。Linux 4090 侧完整镜像位于 `/root/kci_native_4090_20261122/artifacts/history_evaluator_dev_v1/`。

复核脚本：`tools/kci_history_eval_dev_probe_20261122.py`、`kci_history_eval_freeze_pairs_20261122.py`、`kci_history_eval_train_4b_20261122.py`、`kci_history_eval_eval_4b_20261122.py`、`kci_history_eval_nonhistory_control_audit_20261122.py`、`kci_history_eval_finalize_20261122.py`，以及 `code/tests/test_kci_history_evaluator_cell.py`。

**封口状态**：本轮开发格已完成并保存负结果；没有把标签有效性误写成策略收益，没有把派生世界当独立根，没有改动原生 verifier，也没有声称当前算法或 4B 已成功学会历史。
