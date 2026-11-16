# PiSAs 原始加载与评价路径澄清

固定官方 commit：`1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd`。本补充只读检查源码，不运行外部代理代码、不调用模型、不改变既有 `anchors.jsonl` 或源四件套。

**实际默认加载行为已明确，规范协调仍未解决。** 多代理、无 private/shared memory 模式根据 artifact.visible_to 选择接收者，再把整份 artifact.content 加入上下文；没有根据 attribute.visibility 二次删减。属性级 hidden_from 由后续评价器使用。因此本次 11 处差异并非被一个更细 ACL 优先规则自动消除，不能自行选择覆盖规则并声称忠实沿用原系统。

|官方文件和具体位置|观察到的行为|
|---|---|
|[run_pipeline.py L173–186](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/run_pipeline.py#L173-L186)|对 scenario.timeline 调用 preload，传 memory 配置；不传 attribute visibility 表|
|[agents.py L1284–1293](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/agents.py#L1284-L1293)|依据 artifact.visible_to 选择已存在的 agents，并取完整 content|
|[agents.py L1385–1387](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/agents.py#L1385-L1387)|无 memory 的普通分支逐接收者 receive_artifact|
|[agents.py L1052–1054](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/agents.py#L1052-L1054)|整个 artifact append 到该 agent.context|
|[agents.py L66–92](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/agents.py#L66-L92)、[L1099–1124](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/agents.py#L1099-L1124)|原文 renderer 与普通 prompt 构建路径保留 content|
|[run_pipeline.py L214–227](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/run_pipeline.py#L214-L227)|attribute visibility 作为 evaluator ground truth 保存|
|[judges.py L621–637](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/judges.py#L621-L637)、[L678–704](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/judges.py#L678-L704)|从 agent content 抽取属性并核验正确性后，结合 hidden_from 判定可见性违规|

静态微检查记录：11 个属性对应 32 个额外 recipient–attribute 对，其中 4 个属性（14 对）的 utility.value 本身是其整份 artifact.content 的精确子串。余下 Jira 属性拆在 ticket 文本的多行中，不把非连续字符串硬当精确匹配；本次没有调用语义 judge。因而这些数目是源字段/文本暴露的结构检查，不是一次实际 benchmark 的模型违规分数。

Memory 模式需另行区分：源码 [agents.py L1254–1263](https://github.com/ServiceNow/PiSAs/blob/1a47ea461f1f78f2b7a87e38d351e4ebc7380fbd/evaluation/agents.py#L1254-L1263) 表明 private memory 用 LLM 笔记替代原文，共享 memory 可能附加笔记。这里未运行这些模式；不得把普通模式发现泛化成所有 memory 配置的最终观测。Single-agent 的 L1232–1236 原本就加载全部非 task artifacts，也不适用于多代理最小权限假设。

对 KCI 接入的影响：原系统如何预置上下文已可追溯，但“原始初始可见性冲突是否允许、如何作为规范和初始知识条件处理”仍为 UNKNOWN。没有将原函数名中的 preload/deliver 自动提升为 KCI 的实际 A2A 交付证据；`delivered_events` 继续为空，已有 source anchors 继续不可训练。

新增 8 个 pinned Python 文件及其下载记录见 `pisas_github/evaluation/` 与 `download_receipts_clarification.jsonl`；`visibility_clarification.json` 绑定源代码 SHA、原 anchor SHA 和具体差异，微检查 0.027 秒。原始 16 次下载日志逐字节保留，原 `manifest.json` 的既有文件哈希保持有效。

许可表述仅表示本次在所审阅位置未找到项目许可声明，不等于断言研究读取被禁止。当前工作仅为公开来源的只读锚定；未来扩大复用时应另行核实适用许可。
