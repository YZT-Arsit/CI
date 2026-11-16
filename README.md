# KCI-Align

共享执行模型根据动态接收者知识状态，学习即时 contextual integrity（CI）与长期合规任务策略。
研究目标以 [Proposal v0.6](docs/KCI_Align_Research_Proposal.md) 为准。

这是研究原型及 CPU 实验归档，尚无真实 LLM 训练收益或大规模实验结论。

## 当前进度

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

当前完整 153 项回归涉及原始 Windows CPython 3.13 / amd64 的固定 wheel 校验，
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

