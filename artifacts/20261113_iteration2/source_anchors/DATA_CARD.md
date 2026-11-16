# 外部静态来源锚点：PiSAs 与 MultiAgentPrivacy

本次新增 **2 个 PiSAs workflow 样本 + 1 个 MultiAgentPrivacy 关系样本**，只建立来源、真实 schema 和只读导入边界。`anchors.jsonl` 中每条均有 `delivered_events=[]`、`derived_ci_labels=[]`、`dynamic_train_ready=false`、`model_input_ready=false`、`human_verified=false`。没有加入 4286 对偏好，没有生成新的动态 CI 标签、模型调用、上传或训练。

## 固定来源与许可范围

|来源|固定 revision / commit|本次保存|许可证据|
|---|---|---|---|
|[PiSAs HF](https://huggingface.co/datasets/ServiceNow/PiSAs/tree/ac6634013dc74b1c9601d9e59248a59465438a43)|`ac6634013dc74b1c9601d9e59248a59465438a43`|Meeting、Jira 各 scenario/utility/appropriateness/visibility，以及数据卡/API metadata|固定 HF 数据卡明确声明 Apache-2.0；所有场景为 synthetic|
|[PiSAs GitHub](https://github.com/ServiceNow/PiSAs/tree/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd)|`1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd`|README、THIRD_PARTY_LICENSES、完整 tree/commit metadata|审阅 root/README 未发现项目自身 LICENSE 声明；第三方清单不等于项目授权，不将 HF 数据许可外推给代码|
|[MultiAgentPrivacy](https://github.com/Vaidehi99/MultiAgentPrivacy/tree/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41)|`6e7af5d8a8b8a803d8d51cefc01831d2e0724b41`|README、experiment_10、完整 tree/commit metadata|审阅 root/README 未发现 LICENSE/COPYING 或许可声明；训练/再分发复用状态未解决|

公开数据通过官方 HF/GitHub API 和固定 revision 的 raw URL 获取。插件完成仓库及 schema 发现；原始字节下载使用无凭据的官方 URL。每文件独立子进程硬上限 9.2 秒、500000-byte 单文件上限、每来源 5MB 上限。16 次下载均成功，最长 worker 1.347 秒；HTTP 原始保存量分别为 HF 144141 bytes、PiSAs GitHub 12727 bytes、MultiAgentPrivacy 20833 bytes，远低于上限。没有克隆仓库或安装依赖。每个原始文件的 URL、SHA256、ETag、字节数、时间均在 `download_receipts.jsonl`；GitHub connector 的 commit/tree 另存 `connector_provenance.json`。

## PiSAs 两种原始 workflow

|样本|源 scenario ID|初始资料|属性|
|---|---|---:|---:|
|Meeting scenario_01|`meeting_nf_cannot_use_room_s0001_seed20260503`|8 artifacts|22：9 appropriate / 13 inappropriate|
|Jira scenario_01|`jira_config_cannot_be_sole_owner_s0016_seed52424`|8 artifacts|30：24 appropriate / 6 inappropriate|

四文件真实结构：scenario 包含 task/cast/org/timeline；utility 包含 attribute_values、decision_rule、oracle_answer 和 source-specific schedule/assignment 约束；appropriateness 为属性 ID 到静态标签的映射；visibility 为属性的 value/holder/visible_to/hidden_from。全部四件套 ID 和 52 组属性键一致，utility 的原子 value 与 visibility.value 逐项一致。

固定 Meeting 样本的 a9 医疗原因与 a10 房间约束、Jira 的 a22 绩效原文与 a22n 业务约束，确实分别带 inappropriate/appropriate 标签。它们支持静态原文/必要业务约束的来源锚定，不能自动赋予任意新历史下的派生关系 CI 真值。适当性和可见性是不同轴；某人可见一个不适当的原子属性，并不意味着该属性适合用于当前任务。

**可见性解释仍有缺口：** 将 utility.source 指向的整份 artifact.visible_to 与各属性 visibility.visible_to 比较，Meeting 有 3 个属性、Jira 有 8 个属性存在前者额外读者。完整列表在 `audit_summary.json`。这说明接入时不能直接复制整份 artifact 给其声明读者，并同时声称已满足属性级权限；需要核验原始执行器如何分段/筛选。这里记录的是源字段之间的差异，未将其泛化为整个 benchmark 的缺陷。导入器因此不建立任何代理 observation 或接收者已知集合。

全部 timeline 原文只保留在 `initial_materials.initial_artifacts`，没有转成 DELIVERED 消息。utility 的答案、隐藏约束和解释只保留在 `source_task_reference`，没有渲染成模型 prompt。源文件中的时间戳不是本次执行的送达凭据。

## MultiAgentPrivacy 原始关系锚点

[experiment_10](https://github.com/Vaidehi99/MultiAgentPrivacy/blob/6e7af5d8a8b8a803d8d51cefc01831d2e0724b41/data/experiment_10_def_adv_both.json) 实际 schema 是 entities 与 data_distribution[agent].table[table]{sensitivity,rows}，另有 benign/sensitive 目标、组合推断步骤、参考结果、文本计划与 defense 文本。此样本包含 4 agents、8 tables、24 rows。源表级 sensitivity 原样保留，没有提升为带 recipient/purpose 的 CI 规范。

独立只读检查还发现此样本的敏感目标使用 high-income / significant debt，却没有数值阈值；参考只提 John，而同表 James 年龄更高、收入与债务更高且也位于 Urban 城市。因此不能将这段结果直接当“满足条件的完整集合”的精确 oracle。这是该样本的具体歧义，不是对整份数据的统一判定。若未来引入明确阈值与完整集合判断，需要单独版本化为 adaptation 并核验。

## 可复用入口与检查

`code/kci/source_ingest.py` 提供 `load_pisas_anchor` 与 `load_multiagentprivacy_anchor`。调用必须提供完整 pinned revision 和原始文件 SHA256；读操作不修改源文件。导出没有 prompt/chosen/rejected 字段，训练序列适配器明确拒绝这些 source anchor。

`code/experiments/source_anchor_audit.py` 离线验证并导出。7 项边界检查通过：空交付/派生标签、禁止混入训练、SHA 篡改拒绝、跨文件 scenario_id 错配拒绝、缺失属性标签拒绝、同属性可见/隐藏重叠拒绝，以及全部 timeline 保留为初始资料。单 worker 0.269 秒，硬限 9.2 秒。

仍须新增并认证：实际投递账本、观测过滤、动作空间与状态转移、结构化组合推断、派生关系的 recipient/purpose 授权、执行期限与成本、任务终局验证、同状态参考续行和动态偏好证据。两种源 workflow 的存在不能替代这些缺失合同，也不能把 synthetic 原始数据称作真人审核数据或真实组织运行记录。
