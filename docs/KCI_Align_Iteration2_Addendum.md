# KCI-Align 第二轮实现与证据补充

原研究目标和 Proposal/Handoff 不变：共享执行模型在实际接收者知识条件下学习即时 CI 与长期合规任务策略，CCE/DPR 提供监督。本文件记录第二轮的实现入口与证据边界，不替代原研究计划。

完整结果、数据卡和后续门槛见 [第二轮报告](../artifacts/20261113_iteration2/REPORT.md)。上一轮文件保持原样。

## 实现入口

| 文件 | 用途与边界 |
|---|---|
| `code/kci/multi_env.py` | 四种精确推断规则与共享任务流程；canonical source renderer 阻止结构和可见文本矛盾。规范是手工合成合同。 |
| `code/kci/multi_candidates.py` | 有限候选生成；CCE/DPR 仍由 `continuation.py`、`repair.py` 与实验枚举脚本处理。 |
| `code/kci/multi_history.py` | 经公共语法校验的版本化初始历史；不是实际执行数百步的轨迹。 |
| `code/kci/training_data.py` | 严格可见输入、完整回复/EOS、padding、position IDs 与来源/分组检查；尚未接真实模型 Trainer。 |
| `code/kci/batch_evidence.py` | 预声明比较/有限 look 的总体置信预算；可信单账本计账，不提供 anytime 或防篡改保证。 |
| `code/kci/source_ingest.py` | 保存静态来源锚点，显式拒绝将无交付事件/派生规范的锚点当作动态训练样本。 |

在仓库根目录运行回归：`python -m unittest discover -s code/tests`，并将 `code` 加入 `PYTHONPATH`。实验脚本位于 `code/experiments/`，精确参数、种子、源码快照与短 worker receipts 保存在第二轮产物目录。最终回归 99 项通过。

## 数据选择

最终主数据为 `artifacts/20261113_iteration2/dataset_v2/preferences.jsonl`：4,286 个唯一 pair，来自 112 个参数配置、4 个规则骨架和 1 个共同 workflow；96 个观察不可辨识标签在 quarantine 中，不能重新混入。主 split 为 train-only，探索性 family 留出不等于独立任务泛化。

长历史 matched v2 单独保存 140 个唯一 pair，配对侧事件数、字符数和完整 prompt token 数相等。4k 序列预算只能保留 40%；不能通过默默截断或删除长样本声称完成长历史目标。

PiSAs/MultiAgentPrivacy 的 3 个固定版本来源锚点独立保存，未进入上述训练数据。PiSAs 普通多代理、无 private/shared memory 模式下的默认 preload 将 artifact 级整份内容放入 `agent.context`，属性 visibility 用于 judge；详见 [暴露路径附录](../artifacts/20261113_iteration2/source_anchors/ACL_CLARIFICATION.md)。来源数据卡保留早期待核验状态，后续暴露路径结论以该附录为准。

## 当前决策

允许继续有界 CPU 实现与来源审计；**大批量 LLM 训练 NO-GO**。40 个有限共享模型未建立 future/repair 方法收益；真实冻结 Qwen 的 16 回合结果为 2 合规成功、8 违规、2 任务失败、4 UNKNOWN。该结果不等同真实 LLM 参数训练结果。

后续先建立来源支撑、可执行、规范独立验证的不同任务流程及真实交付长历史；再做小规模 GPU 共享 LLM SFT/DPO、匹配消融、数值和长上下文验证。仍需保留负例、UNKNOWN 与来源版本，不以数据行数代替任务独立性。
