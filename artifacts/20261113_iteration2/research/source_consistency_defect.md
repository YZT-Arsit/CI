# 高优先级：公开构造器允许结构事实与可见文本不一致

发现版本：`multi_env.py` SHA256 `8e63919adb6c89be015f0fe2156470c819feb4dc8d4d4aa4eb6b27f41bd9b54d`，`MultiRuleCIEnv` v0.1。独立复现：`source_text_counterexample.py`；完整源值、动作偏好、Prompt hash见同名JSON。

**严重性：高，属于源语义/监督合同缺陷。** 它可以让执行模型收到完全相同的授权输入，却被要求学习相反的严格偏好。不是浮点误差，也不能由更多训练消除。影响公开的显式 `facts=` 构造接口；本轮112个由未改动 `build_case` 工厂生成的正常记录，其文本/值保持一致，未被该外部坏输入直接污染。

## 最小反例

1. 用 `build_case('identity_attribute', answer='yes')` 得到payload：结构值为整数9的字符串，显示文本为 `Record000 has attribute value 9.`；任务条件为threshold=7。
2. 仅把同Fact的 `value` 改成字符串1，保留 `subject`、`text`、Fact ID、任务条件与其它源事实。
3. 将这些事实传给公开 `MultiRuleCIEnv(..., facts=..., task_rule=...)`。v0.1接受它，因为检查了类型/字段，却未检查唯一canonical文本与结构语义一致。
4. 两个环境都执行A SKIP、B SEND payload。B和R的 `Observation.as_prompt()` 逐字相同；同Prompt hash为 `589d5b81f252bae7e39e31807cd5a5a935bd0c10461dfcd863965b05c79ed7c2`（R处）。
5. 正常环境预期答案yes；不一致环境预期答案no。CCE在R处分别导出FINALIZE yes优于no、以及no优于yes的相反严格偏好。

内部环境version发生变化不能解决这个问题：版本元数据不在执行Prompt内，而模型看到的是相同事实文本。训练适配器只检查schema也无法证明文本语义正确。

## 修复验收与数据处理

主代理已决定新增v0.2 canonical source renderer并重跑原112-case计划。验收应覆盖每个source predicate和private roster，验证Fact的subject/value/text一致；对不受支持的表述拒绝或UNKNOWN，不能直接相信结构标签。自由自然语言的独立语义验证仍未建立。

v0.1源码快照、原始数据和本反例保留。v0.2必须使用新的环境version/source hash、重新生成与重放原计划、重跑独立闭包/可达性及全量观测等价审计。不能只改报告版本号或沿用v0.1验证凭据。

## 与另一项可观测性发现的区别

全112-case正常数据另外出现48个同Prompt动作的任务可达性差异，影响96条A处严格通知偏好：隐藏的R任务条件使一处为严格优势、另一处为tie，没有实际反向偏好。该发现不是上述文本注入导致，修复canonical文本后仍预计需要独立隔离。

前者需要修复源语义构造合同；后者需要在声明支持域内限制可学习的严格监督。两者都不要求改变研究目标。
