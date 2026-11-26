# 多源共享策略开发结果

共用一个共享 4B 骨干。每个表项先按原生机会概率计算，再按来源组平均。成本只对安全成功轨迹有定义。scope 衡量证据子任务，temporal 衡量完整业务；协商枚举七种后续，关系推断枚举原生机会，scope/temporal 使用四次共同随机抽样。

|模型|任务|根|STSR|违规|中止|UNKNOWN|
|---|---|---:|---:|---:|---:|---:|
|original|DEAL_OR_NO_DEAL|8|0.000|0.000|1.000|0.000|
|original|MULTIAGENTPRIVACY|7|1.000|0.000|0.000|0.000|
|original|PW_dev|32|0.062|0.000|0.500|0.000|
|original|PW_fresh_dev|32|0.125|0.000|0.562|0.000|
|original|pisas|4|0.250|0.000|0.250|0.000|
|original|relational|1|1.000|0.000|0.000|0.000|
|original|scope|5|0.250|0.000|0.750|0.000|
|original|temporal|5|0.000|0.150|0.750|0.000|
|readout|DEAL_OR_NO_DEAL|8|0.607|0.250|0.143|0.000|
|readout|MULTIAGENTPRIVACY|7|1.000|0.000|0.000|0.000|
|readout|PW_dev|32|0.031|0.031|0.562|0.000|
|readout|PW_fresh_dev|32|0.094|0.000|0.688|0.000|
|readout|pisas|4|0.000|0.000|1.000|0.000|
|readout|relational|1|1.000|0.000|0.000|0.000|
|readout|scope|5|0.800|0.000|0.200|0.000|
|readout|temporal|5|0.050|0.450|0.400|0.000|
|pass2|DEAL_OR_NO_DEAL|8|0.839|0.036|0.125|0.000|
|pass2|MULTIAGENTPRIVACY|7|1.000|0.000|0.000|0.000|
|pass2|PW_dev|32|0.000|0.000|0.781|0.000|
|pass2|PW_fresh_dev|32|0.062|0.000|0.875|0.000|
|pass2|pisas|4|0.000|0.000|0.000|0.000|
|pass2|relational|1|1.000|0.000|0.000|0.000|
|pass2|scope|5|0.450|0.000|0.550|0.000|
|pass2|temporal|5|0.250|0.150|0.600|0.000|
|pass4|DEAL_OR_NO_DEAL|8|0.696|0.018|0.286|0.000|
|pass4|MULTIAGENTPRIVACY|7|1.000|0.000|0.000|0.000|
|pass4|PW_dev|32|0.188|0.094|0.531|0.000|
|pass4|PW_fresh_dev|32|0.312|0.000|0.562|0.000|
|pass4|pisas|4|0.000|0.250|0.000|0.000|
|pass4|relational|1|1.000|0.000|0.000|0.000|
|pass4|scope|5|0.700|0.000|0.300|0.000|
|pass4|temporal|5|0.350|0.200|0.500|0.000|

## 披露与训练目标

FORCED_DISCLOSURE_OPPORTUNITIES.json 逐项记录在同一已执行上游行动与机会条件下的原生结果。DISCLOSURE_BY_DOMAIN.json 汇总可安全成功时的合法披露及任务成功、存在违规备选时的实际违规，并列严格历史匹配分母。ProofWriter 和 PiSAs 的授权保持与历史双侧表现见 HISTORY_METRICS.json。企业多标签要求全部选对；参考排序、合法保持与任务输出各用自身分母。

|模型|来源|目标|根|按根平均正确率|
|---|---|---|---:|---:|
|original|CIWORK|multilabel|8|0.125|
|original|DEAL_OR_NO_DEAL|pair|8|0.240|
|original|MULTIAGENTPRIVACY|pair|8|0.927|
|original|PISAS|hold|6|1.000|
|original|PISAS|masked|2|1.000|
|original|PISAS|pair|8|0.562|
|original|TASK_ONLY|completion|8|0.000|
|pass2|CIWORK|multilabel|8|0.500|
|pass2|DEAL_OR_NO_DEAL|pair|8|0.667|
|pass2|MULTIAGENTPRIVACY|hold|8|1.000|
|pass2|MULTIAGENTPRIVACY|pair|8|0.969|
|pass2|PISAS|hold|8|1.000|
|pass2|PISAS|masked|2|1.000|
|pass2|PISAS|pair|8|0.771|
|pass2|TASK_ONLY|completion|8|0.375|
|pass4|CIWORK|multilabel|8|0.625|
|pass4|DEAL_OR_NO_DEAL|pair|8|0.708|
|pass4|MULTIAGENTPRIVACY|hold|8|1.000|
|pass4|MULTIAGENTPRIVACY|pair|8|0.969|
|pass4|PISAS|hold|6|1.000|
|pass4|PISAS|masked|2|1.000|
|pass4|PISAS|pair|8|0.667|
|pass4|TASK_ONLY|completion|8|0.750|
|readout|CIWORK|multilabel|8|0.500|
|readout|DEAL_OR_NO_DEAL|pair|8|0.344|
|readout|MULTIAGENTPRIVACY|hold|8|1.000|
|readout|MULTIAGENTPRIVACY|pair|8|0.969|
|readout|PISAS|hold|4|1.000|
|readout|PISAS|masked|2|1.000|
|readout|PISAS|pair|8|0.615|

本表的跨任务宏平均为等任务权重；不同通信合同的结果另见逐根表。企业证据选择和业务输出只有静态任务标签，单列于 STATIC_DIAGNOSTICS.json。多源读出使用冻结表示加混合监督评分，不与单域两轴 ridge 混称。既有32根与新增来源开发面板分列。多源分区按本轮来源隔离，过往项目的全部历史曝光不能对每个域独立证明。

本轮采样进入参数更新的唯一记录 597 条，训练四轮。总训练 42.6 分钟。共享读出提取 10,002,285 个输入 token，线性拟合 1.5 秒。

等任务宏平均自主 STSR / 风险为 original 0.366 / 0.021, readout 0.498 / 0.104, pass2 0.506 / 0.027, pass4 0.562 / 0.080。该单种子开发比较保留全部负结果。

## 同一 32 根的最终研发比较

原菜单策略实际评分正逆两个共同顺序。多源模型逐个读取语义候选，按固定次序组成推理批次，外部菜单反转映射到同一组候选分数。多源起点与菜单起点使用同一检查点，接口不同，分别列出。

|模型|历史联合正确|合法机会成功|合法披露|危险侧违规|授权保持|自主 STSR|自主违规|中止|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|initial|0|7|22|21|10|12|1|6|
|S1-2|0|9|11|15|16|11|2|18|
|S1-4|0|11|11|15|22|12|0|20|
|S1-8|2|9|11|10|22|8|2|21|
|s2_action|3|12|14|11|21|10|2|19|
|s2_process|0|11|11|15|21|11|1|18|
|s3|3|20|25|23|23|15|6|6|
|grpo|0|21|29|27|24|16|7|4|
|n128|3|6|23|10|2|6|2|9|
|n128_lam_0.0001|4|8|24|10|8|8|2|11|
|n128_lam_0.01|0|11|27|10|17|9|2|4|
|n128_lam_1.0|0|11|31|13|12|5|7|1|
|n16|1|13|17|12|8|6|2|15|
|n64|2|7|25|14|3|10|3|8|
|multi_original|0|3|18|7|8|2|0|16|
|multi_readout|0|2|13|4|9|1|1|18|
|multi_pass2|0|1|6|2|11|0|0|25|
|multi_pass4|2|7|13|6|18|6|3|17|

## 下一阶段裁决

共同开发面板上的多源候选均未同时提高严格历史联合正确和自主 STSR 并保持观察风险。保留充分训练的动作对照；本轮读出和过程蒸馏尚不足以进入正式主表或扩大种子。
四个来源的完整任务宏平均仅包含实际自主执行。相同来源根的多种任务先在根内平均，再等权平均来源根和来源域。静态标签、scope 证据子任务及独列的新来源 ProofWriter 面板不进入该分母。详见 FULL_TASK_SOURCE_MACRO.json。
分域训练与行为证据用于下一轮同一动态 CI 问题的误差定位。封存最终测试集继续保留。
