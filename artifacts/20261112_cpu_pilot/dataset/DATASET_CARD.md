# KCI CPU synthetic v1 数据卡

用途：有限环境回归、证据链/训练管线诊断。禁止据此宣称真实CI、独立世界泛化或方法收益。

来源：本项目原创建模数据，无PiSAs/MultiAgentPrivacy样本复制，无人工规范真值。2016条原始记录，1824条唯一表面对，38种canonical偏好模式。48组名称/编号/eligible薪资带只是同一基础世界的变体；基础组统一toy_eligibility_join_v1，规则家族identity_attribute_join_v1。没有独立validation/test。

`preferences.jsonl`保留prompt/chosen/rejected及环境配置、实际前缀、直接/增量审计、参考见证、版本、来源成员、原始文件行位置和哈希。`canonical_preferences.jsonl`是38种结构代理输入，保留对应原记录ID与重放配方，不是独立任务集合。`shards/*.verification.v2.json`绑定原始字节SHA256与最终core；失配会阻止assemble。

来源计数：CI96、future1344、repair576。CI与repair重叠96，future与repair重叠96；不能把交叉成员当额外独立样本。最终全局精确同Prompt反向pair扫描为0；另有隐藏背景反例被保存在batch_integrity并隔离，说明零冲突不保证所有未来数据都可观测。

规则限制：精确事实ID、canonical消息、单identity-attribute join、固定eligible=yes。全部human_verified=false、natural_language_semantics_verified=false。许可/治理：这里仅声称原创建模；引用论文PDF不是本数据的标签授权或隐私证明。

生成代码：code/experiments/build_dataset.py；合并代码：assemble_dataset.py；独立结构审计：corpus_audit.py。完整配置和版本见相邻summary及上层input_manifest和final_manifest。
