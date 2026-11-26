# KCI 多来源、多角色训练数据 v1

本包用于内部研究。训练样本绑定在执行角色当时可见的观察及动作空间上。一个来源根可以产生多个角色、历史、授权条件和机会事件。统计独立性时使用 `group`，统计优化曝光时另计训练记录与完成 token。

## 数据入口

| 文件前缀 | 训练行数 | 学习对象 |
| --- | ---: | --- |
| train_sft | 79,521 | 完整原生语义动作。接受集合中有多个动作时，每个 epoch 轮换示范。 |
| train_dpo | 4,062 | 同状态下的完整 chosen/rejected 语义动作。可按监督来源和价值维度筛选。 |
| train_task_sft | 1,160 | 完整业务答案、已接受的终局动作、经验证的多角色成功路径动作。 |
| train_task_dpo | 69 | 普通任务准备、完成对中止等有方向的业务任务偏好。 |
| train_static_evidence | 84 | CI-Work 静态条目选择。目标输出包含应选条目 ID 的完整 JSON。 |
| train_process_sft | 1,715 | 已有 ProofWriter 私有过程与其支持的具体行动，仅按显式选择加载。 |

行数为去重后可被优化器读取的记录。六个训练分片共 86,611 行。各行的 `actor_context`、`executable_actions` 构成学习器输入。目标及来源字段留在模型输入外。一个集合标签表示多个动作均可用。单条成功续行只证明该动作在归档的续行下成功。冻结参考策略下的优选动作也可能有多个合法替代。任务阶段、规范范围、分支状态等原始证据通过 `record_id` 与 `provenance.jsonl.gz` 对接。

## 来源与准入

| 来源 | 本版采用的证据及用途 |
| --- | --- |
| D_future_core_multisource_v2 | 核对总表和 SHA-256。3,559 对中，2,664 对进入训练，800 对归入历史开发，95 对归入本版按组留出。训练部分的 2,499 对安全成功与 165 对条件保护成本使用 `value_tiers` 区分。 |
| KCI_VALIDATED_V9_INTERNAL | 按原始 ACL、合成即时 CI、真实失败、参考续行、终局、任务准备及静态证据分别接入。242 条原诊断参考记录中，有 199 个 ID 在新版核心交付中重新通过验证，只走新版核心入口。V8 的 195 条语义重复、旧业务研究的 29 条顺序混杂记录以及待审规范不作为训练来源。 |
| Deal or No Deal | 对通过独立校验的 1,837 个上游偏好实例，复核两条签收草案之后的七种事件和调解者完整菜单。训练 SFT 含调解者决策 21,966 条。上游 A 的偏好另走 DPO。 |
| ProofWriter | 从 1,265 个实例和 23,780 个联合评价单元恢复角色输入。原生执行器补算归档对照之外的分支。区分原始报告和允许本地证明的合同，保留授权条件与安全退出。训练 SFT 含 A 行动 1,420 条、B 行动 4,911 条。B 输入仅包含允许的规则、自己的持有事实、已交付消息与自己的菜单。 |
| PiSAs、关系推断与聚合 | 复用已执行的安全成功续行，以及中继与临时授权原生训练档案。按状态重放关系推断、委托人和执行者动作。聚合第二分析者仅看到自己的队列与公共任务，示范动作必须在相同可见观察对应的不同前史上均可安全完成。归档固定参考的成功见证、合成信息流规范和原始 ACL 保持独立标记。 |
| 业务与较早训练资产 | 收入业务补充包的源数据任务、经核验的 Phase 2 完整团队示范、DG 原生任务对、V4 独立审计通过的同状态训练对、经协议等价审查的 8 条有限工具语言变体。文字改写、菜单反序和机会分支沿用原组。 |
| 历史模型失败 | V9 的既有真实失败对与 ProofWriter 学生实际访问状态分别保留来源。其中后者有 194 个可训练动作对，含同一失败对应多个可行修正的情况。可追溯到采样快照。 |
| 仅作控制、待审、外部基线 | 494 条 V9 控制放在单独文件。业务补充的控制索引、TOP-R 待审规范、未审 API 候选、工具单测示例、PrivAct/CDI 的原项目输入和 CI-Repair 的独立协议均保留原位置。现有材料没有把它们统一认证为本任务的角色行动教师。 |

在训练 SFT 中，PiSAs 有 50,186 条，协商有 21,966 条；不同来源的机会枚举规模差异很大。训练端一轮可完整遍历指定分片，逐组轮转会在长尾阶段继续消费剩余记录。报告源组宏平均与优化曝光量时需要分别计数。全部公开规范与合成规范的有效范围仍以原交付为准。

## 分组与评价

| 分区 | 来源组 | 口径 |
| --- | ---: | --- |
| train | 881 | 允许用于梯度。 |
| validation | 40 | 本次发布预留的 4,133 条有标签记录。与此次训练按来源组及完整角色观察隔离，可用于跨组开发评价。 |
| development | 254 | 既有开发或诊断分区。 |
| development_exposed | 1 | 含既往拟合及诊断暴露的组，保留作复算。 |

本版留出覆盖协商 22 组、PiSAs 8 组、CI-Work 5 组、MultiAgentPrivacy 3 组、ProofWriter 1 组与 TOP-R 1 组。各条目及原生来源行见 `VALIDATION_INDEX.json`。静态条目选择和任务子目标仅支持各自的离线指标。自主 STSR 仍须通过模型选择动作、执行器实际交付及终局审计计算。

`SPLITS.json` 记录原始源组、可见估值组、ProofWriter 原文与理论别名、原文件指纹、历史训练与开发分配。24 个前瞻 PiSAs 来源只登记元数据。最终集保持封存。本次留出不能排除外部预训练或未登记的历史接触，ProofWriter 单组结果尤其不宜作正式泛化结论。

## 训练接口

本地入口为 `tools/unified_data/train.py`，GPU 便携包内为 `train.py`。以完整接受集训练时使用 `--route sft`。例如，`--route sft --tier native_proof_B --role source_B` 读取 ProofWriter 下游监督。`--role` 根据当前执行者与任务中声明的持有者、委托者、中继和执行者映射筛选。`EXECUTION_ROLE_COUNTS.json` 给出各角色实际计数。历史模型修复使用 `--route dpo --tier train_actual_failure_dpr` 或 `--route dpo --tier student_failure_repair`。下面的例子只取新版核心参考中的安全成功维度，输出必须使用一个新目录。

```bash
python train.py --data . \
  --base /root/autodl-tmp/kci_models/Qwen3-4B-Instruct-2507-ms \
  --adapter /root/autodl-tmp/kci_v9_training_18828/proof_cce_7/adapter \
  --route dpo --tier core_reference --value-tier SUCCESS \
  --epochs 1 --output ./run_core_success
```

动作和任务 SFT 只对完整 completion 及其结束 token 计算 CE。DPO 比较完整语义动作序列，使用单独冻结的同起点参考适配器。DPO 的参考模型对数概率和环境教师的续行价值分别计算。当前执行模型的实际续行价值不在此数据包内。调用 `examples(data, route, tier, value_tier)` 可直接取得训练流。过程分片中的 `private_record` 仅供角色本地使用，接入原生执行器时只交付 `action`。

## 复核

在研究工作区运行以下命令可重新构建、重用本地已验证分支缓存，并独立核验整个输出。原始数据仍位于各自的只读交付目录。

```powershell
python tools/unified_data/build.py
python tools/unified_data/audit.py artifacts/20261125_unified_roles_v1
```

`MANIFEST.json` 给出源文件与分片哈希，`COVERAGE.json` 给出分角色、目标和来源的计数。`AUDIT.json` 记录逐行复核、跨分区碰撞及消费接口验收。记录数和合法性见证都不衡量学生的训练收益或实际违规率。

