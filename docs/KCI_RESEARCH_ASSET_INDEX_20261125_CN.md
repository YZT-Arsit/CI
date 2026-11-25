# KCI-Align research evidence index / 2026-11-25

本文件是只增不改的研究索引；不是新论文实验结果。研究问题保持为接收者知识累积下的多 Agent CI 学习。原始目录、数据、权重、分区和阴性结论均保留。

## 1. 已冻结的科学证据
| 命题 | 证据入口（相对 D:/research/longhistory） | 裁决 |
|---|---|---|
| 当前历史改变披露合规性 | docs/KCI_HISTORY_EVALUATOR_DEV_20261122_CN.md; docs/KCI_V4_SEMANTIC_CONTROL_20261123_CN.md | 可重放历史对成立；旧 4B 未学会双向 Flip |
| 当前均合法的行动改变未来合规可达性 | artifacts/20261125_kci_v08_verification/Q1_SIGNAL_SUMMARY.md | 14 个签名包 A 状态中 4 个严格原生长期成功差，3 类规则；授权消除差异；不把 38 个非严格即时等价反转误报为严格反转 |
| 当前策略能力影响续行标签 | artifacts/20261125_kci_v08_verification/BASELINE_RECONCILIATION.md | K8 接受 0/14 到 7/14；正式 adaptive 确认方向与匹配终局教师 7/7 一致 |
| DPR 的失败覆盖 | artifacts/20261125_kci_v08_verification/Q1_SIGNAL_SUMMARY.md | 42/47 原失败观察不在成功轨迹中；同错题纠错复现所有修复 |
| 共享策略更新退步 | artifacts/20261125_kci_v08_verification/Q2_POLICY_IMPROVEMENT_REPORT.md | 数学反例、PDL 和历史 CPU 退步成立；新投影与普通选点均 0/3 非零改善 |
| 多任务学习证据 | docs/KCI_V8_MULTITASK_CONFIRMATION_20261124_CN.md | 冻结基座到共享模型独立业务 STSR 22.1%→49.3%；企业证据监督 4/15→10/15；不是 CCE/DPR 增益 |
| 论文原生实现 | docs/KCI_PAPER_NATIVE_VERIFICATION_20261124_CN.md | 有限动作环境、原生重放和真实 4B 消融；增补臂安全成功增加同时出现组合违规 |
| 新领域真实模型试验 | docs/KCI_V9_QWEN3_ALGORITHM_PILOT_CN.md; docs/KCI_V9_HISTORY_DEPENDENT_DPR_CN.md | ProofWriter 16 个 DEV 根：SFT/终局 DPO 各 9/16，CCE 6/16；菜单交叉后联合正确均 0/32；不重复原五臂 |
| GPU 可用性 | artifacts/20261125_v08_phase3_gpu/GPU_READINESS.md | 预检 PASS；正式 v0.8 Phase 3 NO_GO |

## 2. 数据资产与权限
- D:/research/kci_supervision_alignment_20261123/deliverables/KCI_VALIDATED_V9_INTERNAL：2405 条分层唯一记录（含控制与诊断）；不是全量可训偏好。保留 typed loader、29 条旧记录 denylist、分组和许可限制。
- D:/research/kci_supervision_alignment_20261123/d_future_core_20261124/D_future_core_multisource_v2：3559 条条件参考偏好、940 个来源/可见分组；不能与 V9 简单相加；非目标模型 Qpi，派生规范未全部人审。
- 新提取：artifacts/20261125_kci_continuation_table_v1/SUMMARY.json；1265 个 ProofWriter 行、473 组、23780 单元、591 行两首动作覆盖完整首步菜单。其余首步未标注项 UNKNOWN，不能作为负样本。
- 新提取仅复用 source_train 归档；不打开 DG36/P211/P212 或其他锁定集合。训练前排除此前开发、独立评估及测试涉及的源组；不要声称全部 473 组均为未见。
- 原始归档中的计算、证明和标签均属于 evaluator-only，严禁注入执行模型 prompt。

## 3. 冻结/推进决定
1. 保留现有 CCE、DPR 作为验证器及经验工具；当前独立算法增量未建立，不重复同预算同标签实验，不启动 v0.8 四臂。
2. 新候选学习机制使用已验证的首步-后续动作联合评价，直接学习知识条件化续行效果，而非再扩大二元偏好采样。执行入口见 docs/KCI_CONTINUATION_TABLE_PILOT_20261125_CN.md。
3. 所有新实验单独保存；研究结论按问题证据、方法、泛化分开，最终封存集维持原样。规范盲审与来源许可证另行完成。
