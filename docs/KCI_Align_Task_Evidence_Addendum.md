# Task evidence 修复与 CCE 合同补充

本补充不改变 Proposal v0.6 的研究目标。当前工程实现为
`dynamic_tasks_v2_evidence`，历史审计使用原样保存的
`legacy_dynamic_tasks_v1.py`。

Meeting 的预订计算必须消费 HR 交付的业务约束；Room 只可直接读取显式
授权的公开排期数据。Query 的 JOIN 必须消费交付的 counts/cities，公开
公司名称读取属于 JoinTool 的显式权限。投影、结果转发及最终交付均消费
消息，不以全局结果缓存代替输入。统一 TEDT 覆盖缺失、篡改、阻断、值变化、
缓存投毒和源权限撤销，来源可追踪至最终 Client 输出。

严格 CCE 保留 `CI > 合规成功 > 成本`。任意原始 LLM 的风险严格等价不能
由有限零违规样本推出。可选的执行网关证明安全转移不变式；在有限精确
可达性成立时进一步证明成功，允许比较实际执行参考策略的成本。
该参考是“LLM + 网关”，网关介入须单独计数，不能当作模型学习成功。

当前 CPU 证据仍不足以批准 GPU 训练：任务成本单位会改变路线排序，
移除成本/紧迫期限后初始路线同值；结构化图分类器的成功尚未转化为
共享文本执行策略的跨实例泛化。数据 `gpu_training_eligible` 明确为 false。

详细结果、原始记录与剩余卡点见
[本轮报告](../artifacts/20261115_dependency_repair/REPORT.md)。
