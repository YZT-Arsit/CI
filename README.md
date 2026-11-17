# KCI-Align

共享执行模型根据动态接收者知识状态，学习即时 contextual integrity（CI）与长期合规任务策略。
研究目标以 [Proposal v0.6](docs/KCI_Align_Research_Proposal.md) 为准。

这是研究原型及 CPU 实验归档；已进行真实预训练语言模型的 CPU LoRA 学习诊断，尚未证明完整 KCI 方法收益或大规模实验就绪。

## 当前进度

最新续轮：[少数状态加权与行动损失诊断](artifacts/20261117_fallback_weight/REPORT.md)。相同初始检查点、批次和160次总更新，只改变无解状态的行动监督权重。新DEV中冻结初始/普通续训/加权续训分别成功52/56/50个（每组112个），违规12/11/9个；加权牺牲效用，未修复位置捷径。新无解状态三模型均为0/12通过全排列。324条条件概率诊断中，集合概率与贪心选择的差异仅出现在各模型违规选择中的0/54、1/50、1/53，不支持作为主要错误模式。**局部修复门槛未通过，完整目标未完成，GPU大批量NO_GO。** 恢复原始记录：`python -B tools/restore_fallback_weight_artifacts.py`。

以下保留上一轮结果：

最新续轮：[续训局部改善与无解时的选项绑定问题](artifacts/20261117_optimization_repair/REPORT.md)。新增两个检查点、160 次更新；同一新 DEV 上严格成功从 43/112 增至 54/112，合规从 92/112 增至 94/112、95/112，但深层完整配对仍全失败。冻结排列干预发现：无合规完成路径时，仅将危险 Full 放到 A/B/C，违规就变为 0/36、18/36、36/36；有解对照全部通过。另完成 [84 次真实条款依据探针](artifacts/20261117_grounded_norm_probe/REPORT.md)，明确 PASS/FAIL 全对，但 UNKNOWN 全部被误判为确定结论。**完整目标未完成，GPU 大批量 NO_GO。** 四组原始归档恢复：`python -B tools/restore_optimization_artifacts.py`。

以下保留上一轮结果：

最新续轮：[冻结模型交叉：关系判断如何影响行动](artifacts/20261117_frozen_crossover/REPORT.md)。三个冻结检查点在 560 个相同状态呈现上交叉测试，未训练新模型。正确输入 SFT 自身判断的严格成功为 39/112，输入环境真值后为 104/112；固定行动模型的干预修复 92 个状态动作错误，但 oracle 下 Worker 全状态仍有 46/224 个错误。关系判断能进入行动，可靠自预测与安全行动仍未解决。主实验 11,392 条生成响应、1,344 个重构回合；训练拟合补充单独记账。**完整目标未完成，GPU 大批量 NO_GO。** 原始记录恢复：`python -B tools/restore_frozen_crossover_artifacts.py`。

以下保留上一轮结果：

最新续轮：[正确训练输入与行动学习对照](artifacts/20261117_teacher_binding/REPORT.md)。新增两个 CPU LoRA 检查点、160 次更新，五模型共 560 回合通过原生审计；纠正行动训练输入仍未解决联合要求，普通监督仅增加 1 个成功回合，关系一致性版本减少 21 个成功回合并减少 6 个违规回合。较早参考一跳仍为 8/8，但总体有 29 个违规回合。另接通了[原生模型协作探针](artifacts/20261117_native_agent_probe/REPORT.md)，与角色说明修复合计 36 次请求、复用 1 个任务，成功工具调用为 0；发现并补查了“初始状态就通过、但缺少转人工动作”的评分缺口。**完整目标未完成，GPU 大批量 NO_GO。** 恢复本轮三组归档：`python -B tools/restore_teacher_binding_artifacts.py`。

以下保留上一轮结果：

最新续轮：[一致性约束拆分与强参考对照](artifacts/20261117_value_consistency/REPORT.md)。新增两个 CPU LoRA 检查点、160 次更新，主实验 448 回合及单独登记的旧检查点参考 112 回合均通过原生复核。去掉动作一致性后一跳完整配对从 0/8 回升至 4/8，价值类版本为 2/8；更早普通监督仍为 8/8，深层关系均未通过，安全—效用联合要求仍未满足。外部规范主探针及位置诊断共 128 次请求、复用 32 个情境，模型始终语义弃答，未建立自然规范迁移证据。**完整目标未完成，GPU 大批量 NO_GO。** 外部结果见[规范探针报告](artifacts/20261117_external_norm_probe/REPORT.md)，归档恢复使用 `python -B tools/restore_value_consistency_artifacts.py`。

以下保留上一轮结果：

最新续轮：[关系进入行动：采样修复与完整对照](artifacts/20261117_sampler_repair/REPORT.md)。固定状态诊断发现行动绑定失败；补齐原训练遗漏的 8 组关键关系监督后，新训练两个 CPU LoRA 检查点并完成四模型共 448 回合原生复核。一跳完整配对成功：普通监督 8/8 → 1/8（Planner 未退步，下游停止增加），一致性模型仍为 0/8；深层关系未通过。新增 493 条 PrivacyLens 外部规范评估候选，固定来源与许可，未用于训练、尚未完成模型评估。**修复方案尚未有效，GPU 大批量 NO_GO。** 数据与重放见[复核说明](artifacts/20261117_sampler_repair/REPRODUCE.md)。

以下保留上一轮结果：

最新续轮：[实体换名、选项绑定与一致性修复审计](artifacts/20261117_binding_factors/REPORT.md)。完整因素交叉区分了上游关系估计错误与下游行动绑定错误；新增两个 CPU LoRA 对照检查点，共 160 次更新，单次最长 5.482 秒。普通两视图监督在新 DEV 一跳历史配对、两种呈现下为 8/8，但深层关系未过；增加条件一致性的方案退化至 0/8，并出现过度停止，**该修复配置失败，GPU 大批量仍为 NO_GO**。原生角色隔离接口通过 114 个任务的执行检查，但模型决策为 0，不能称为 32 步协作。数据、负结果与边界见[复核说明](artifacts/20261117_binding_factors/REPRODUCE.md)。

以下保留此前结果：

最新报告：[关系判断进入行动：CPU 修复与审计](artifacts/20261117_action_binding/REPORT.md)，通俗方案见 [SOLUTION.md](artifacts/20261117_action_binding/SOLUTION.md)。
同一语言模型的自预测关系判断已在有限一跳任务中显示对行动的因果作用；补齐等价状态监督并平衡查询标签后，一跳完整回合得到 8/8 配对成功，但更深关系和部分安全条件仍失败。**GPU 大批量 NO_GO，完整研究目标尚未完成。**
外部 τ²-bench 已完成 114 个标注计划的原生执行核验；这不是模型自主协作，也没有达到 32 步。GitLab 官方规范已固定来源并建立 28 个条款级模型审查案例，跨任务有效性仍未验证。187 项回归通过，失败结果与审计边界均保留。
原始记录、检查点和复核命令见 [复跑说明](artifacts/20261117_action_binding/REPRODUCE.md)；运行 `python -B tools/restore_action_binding_artifacts.py` 恢复本轮无损压缩的 JSONL。

以下保留上一轮结果：

上一轮报告：[关系泛化与安全—效用修复](artifacts/20261116_relation_solution/REPORT.md)，通俗方案见 [SOLUTION.md](artifacts/20261116_relation_solution/SOLUTION.md)。
采用外部任务计算与自建配对干预，建立 1,152 个训练世界、5,760 个状态和 13,824 个动作标签；新增 8 个外部会议实例均核验原始计算结果。
修复 Source 任务依赖编码与实体表示误匹配后，结构参考模型在新的 3,360 个完整条件回合中全部符合各自验收标准，原生输出迁移 384/384；它仍具有明确提供的组合结构和局部关系监督。
本轮训练了 9 个 SmolLM2-135M CPU LoRA 检查点，语言模型仍未通过关系学习门槛，**GPU 大批量 NO-GO**。180 项回归通过；直接学习的不稳定种子、证明幻觉和所有负结果保留。
数据、原始输出、模型与审计见 [复跑说明](artifacts/20261116_relation_solution/REPRODUCE.md)；先运行 `python -B tools/restore_relation_artifacts.py` 恢复本轮 JSONL。

以下保留上一轮结果：

上一轮报告：[无成本任务依赖、原生输出对照与关系学习验证](artifacts/20261115_semantic_repair/REPORT.md)。
当前仍为 **GPU 大批量训练 NO-GO**。本轮完成 69 个小型策略检查点、94,368 个主评测回合与 2,688 个原生输出迁移回合，173 项回归通过。
任务层面的条件性价值差异已能在零成本、无截止时间下成立；原生输出对照无需新增凭据字段。
图策略基础历史反转配对三种子均为 128/128，但长链与无解状态仍失败；token 策略尚未学会可靠关系，强化安全后出现拒答/删减退化。
这些是有限合成规范下的 CPU 证据，不是预训练 LLM 收益或完整 GPU 放行证明。各版本及失败结果均保留。
本轮小型策略检查点已归档；JSONL 使用 SHA256 验证的无损 gzip。先运行 `python -B tools/restore_semantic_artifacts.py`，再按 [复跑说明](artifacts/20261115_semantic_repair/REPRODUCE.md) 操作。

以下为上一轮记录，保留其历史结论：

最新报告：[消息依赖修复、严格 CCE 与关系学习审计](artifacts/20261115_dependency_repair/REPORT.md)。
当前判断仍为 **GPU训练 NO-GO**。Meeting/Query 已实际消费必要消息，统一 TEDT 通过，
完整回归 153 项通过。新增执行网关使严格 CCE 可用于明确受控的参考策略；
Qwen 16 个分支续跑成功，但只有 6 个全程无需网关替换，不能据此宣称模型学会了规律。
三跳图分类器通过辅助诊断；原文本策略的符号泛化仍未证明，成本单位变化也会改变路线排序。
原研究目标不变。工程修复不等于通过完整研究验收。

[此前 GPU 前提审计](artifacts/20261115_relation_gate/REPORT.md)及其负面证据保留：
历史文本模型严格等长新符号测试 384/768、配对两侧同时正确 0/384。
消息旁路已在本轮修复，不能据此改写旧模型的泛化结论。

以下为[第三阶段研究报告](artifacts/20261113_stage3/REPORT.md)的历史结果，不能单独作为当前GPU就绪依据。

| 项目 | 已验证结果 | 证据边界 |
|---|---|---|
| 动态任务 | Meeting、Query 两个可执行流程 | 两个源实例；共用三关系 join 隐私语法；规范为合成改造 |
| 监督数据 | 208 条唯一偏好；8 条历史条件化主监督、4 组配对 | 非独立任务泛化证据 |
| 分层 CCE | 192 项比较：74 个偏好、118 个 UNKNOWN | 下层比较需要可验证的上层等价证据 |
| 有限模型 | 短期限配对正确数 6/12 → 10/12 | 三种子、小型词特征模型；非 LLM 学习结果 |
| 冻结 Qwen | 20 次定点请求；历史配对两侧同时正确 0/4 | 后续为规则参考续行，非完整 Qwen 回合 |
| DPR | 158 个可修复状态，独立新增偏好 0 | 不支持额外方法收益 |
| 本地回归 | 124 项测试通过 | 包含固定 Windows tokenizer 依赖合同 |

消息依赖门槛已在当前有限任务合同内通过；真实训练前仍须通过策略关系泛化和业务成本依据门槛。
独立任务留出、规范验证和 GPU 长上下文测试仍待完成。

## 阅读与数据入口

- [当前修复后的数据与主监督](artifacts/20261115_dependency_repair/training_steps_v2_final/summary.json)：198 条唯一偏好、8 条主监督，明确禁止解读为 GPU 就绪。
- [执行器与 CCE 合同补充](docs/KCI_Align_Task_Evidence_Addendum.md)
- [训练信号覆盖](artifacts/20261113_stage3/SIGNAL_COVERAGE.md)
- [任务合同与来源边界](artifacts/20261113_stage3/TASK_CONTRACTS.md)
- [历史主监督](artifacts/20261113_stage3/training_v3/history_primary.jsonl)
- [全部训练偏好](artifacts/20261113_stage3/training_v3/all_preferences.jsonl)
- [机器结果](artifacts/20261113_stage3/results_summary.json)与[就绪性](artifacts/20261113_stage3/readiness.json)
- [第二阶段报告](artifacts/20261113_iteration2/REPORT.md)、[首轮报告](artifacts/20261112_cpu_pilot/REPORT.md)
- [原型接口说明](code/README.md)：该文件保留早期审计状态，当前结论以最新修复报告为准。

## 快速运行

核心任务/CCE测试仅需 Python 3.10+，从仓库根目录执行：

```sh
cd code
python -B -m unittest discover -s tests -p "test_dynamic*.py" -v
python -B -m unittest discover -s tests -p "test_stream_cce.py" -v
python -B -m unittest discover -s tests -p "test_stratified.py" -v
python -B -m unittest discover -s tests -p "test_task_evidence_dependency.py" -v
python -B -m unittest discover -s tests -p "test_execution_envelope.py" -v
```

重新生成动态任务数据（输出到新目录，避免覆盖研究归档）：

```sh
python -B -m experiments.dynamic_signal_dataset ../artifacts/local_replay ../artifacts/20261113_iteration2/source_anchors
```

上一轮完整 153 项回归涉及原始 Windows CPython 3.13 / amd64 的固定 wheel 校验，
新增关系探针还需 NumPy（验证版本2.5.2；建议在独立虚拟环境安装）。
在兼容环境中，从仓库根目录先恢复依赖资产，再运行：

```sh
python -m pip install numpy==2.5.2
python -B tools/prepare_tokenizer_assets.py
cd code
python -B -m unittest discover -s tests -v
```

恢复脚本只下载清单指定的 tokenizer 和三个依赖 wheel，校验 SHA256，安装至仓库内隔离目录。
它不下载模型权重、不修改系统 Python。其他平台应先建立单独的依赖清单并复验，不能沿用 Windows 资产证明。

## 本次上传范围

这是可审阅的研究快照，不是整台机器的备份：包含当前代码/文档、消息依赖修复的数据与实验记录、最终第三阶段结果、关键审计、任务源锚点和完整测试所需的旧偏好输入。
本轮大体积 CCE 原始记录采用逐字节核验的无损 gzip，见修复报告中的压缩清单。
第三阶段早期数据/模型版本、重复源码快照、论文 PDF、模型权重、依赖缓存及大量旧轮次中间文件留在本地。
报告中的部分历史索引因此指向本地完整归档，不代表这些文件已上传。

[UPLOAD_MANIFEST.json](UPLOAD_MANIFEST.json) 是本仓库本次上传的逐文件 SHA256 清单；
原实验 `final_manifest.json` 保留完整本地实验的覆盖范围，不能作为本仓库的文件存在性清单。
原始报告、数据和日志字节不作改写，日志中的本地绝对路径是历史运行记录，复跑时应改为当前路径。

## 来源与许可

第三方来源及固定版本见 [source data card](artifacts/20261113_iteration2/source_anchors/DATA_CARD.md) 和
[许可核查](artifacts/20261113_iteration2/source_anchors/ACL_CLARIFICATION.md)。
PiSAs 数据卡的许可不自动覆盖其独立实现；固定 MultiAgentPrivacy 源未找到许可文件。
这些锚点用于当前私有研究仓库的追溯，不据此授予再分发许可。
未对用户代码擅自添加开源许可证；如以后公开仓库，应先确认代码和第三方材料的发布范围。

