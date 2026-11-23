# KCI Align v0.7 开发交付

本轮修改了可执行的监督生产与准入流程，没有另开模型训练或重生成数据。目标是阻止“新观察但同标签”再次进入 CCE 方法训练，同时保留有真实增量时进入单独公平比较的路径。三阶段研究尚未全部完成：可靠执行器、新独立根长期监督、最终方法收益都没有被本次代码开发替代。

## 已实现

1. `tools/kci_unique_supervision.py`：逐状态比较成功路径 CE、终局均值、同标量置信过滤、CI-first 点估计和 CCE；分开未知证据、同向、弃权、CE 冲突和新增覆盖。独立确认需要两个固定策略各自的发现/验证块，不以新观察数量冒充增量。
2. `tools/kci_cce_discovery_run.py`：新冻结协议包含新增模块、能力文件和费用文件的哈希；分析实际调用准入模块。能力检查绑定 checkpoint，检查条件完成、分阶段能力、稀有路径和合法 RAW；费用缺项为 UNKNOWN。失败时 `CONFIRMED_LABELS.json` 为空，额外写入逐状态 `UNIQUE_USEFUL_SUPERVISION.json`。
3. `tools/kci_cce_discovery.py`：保留原统计规则，新增每动作风险/合规完成估计、成功条件成本和 CI-first 点估计对照。没有为了得到偏好而改变阈值、成本、标签或 verifier。
4. `tools/kci_cce_three_arm_A_train.py`：旧同标签七状态实验默认在加载模型之前退出。只可通过 `--legacy-negative-control` 显式复现历史负对照，不得用于方法收益主张。
5. `tools/kci_supervision_replay.py`：将冻结 Linux 产物按原哈希绑定本地 TRAIN 教师文件，核验状态池来源，逐步重放真实分叉，检查当前合法性、状态/观察摘要、原生判决和终局。缺少参考策略时保留缺失，不借用采集策略冒充参考。
6. Proposal 已更新 v0.7；ACL 替换段落见 `KCI_Align_ACL_v07_Revision.md`。仓库中只有九页原 PDF，没有找到可编辑排版源，因此未声称原 PDF 已重新排版。

## 准入含义

当前实现一条保守的 UUS 路径：记录的 CE 未覆盖该观察；CCE 的长期方向跨策略与独立块确认；终局均值没有稳定重复同一方向；条件能力和完整费用通过。该门通过后也只是允许另行冻结比较协议，不自动训练。

若差异来自 CI-first 与标量目标不同，结果显式记录 `objective_matched_advantage_established=false`，下一次必须保留相同分叉上的 CI-first 点估计对照。相同目标、相同样本上的点估计往往同向，不能用“CCE”名称把这些信息算作新算法。

对已有 CE 的可信纠错、低预算监督可靠性优势和覆盖—费用曲线是另外几条可能的研究假设，本轮没有实现为自动放行路线，也没有把缺失费用填零。成功见证不同于 Qπ 的排序；分歧不能直接记为错误纠正。这里的 UUS 只相对登记的教师库存，不证明其他教师理论上无法获取标签。

## 真实数据运行结果

正式结果：`artifacts/20261123_v07_supervision_pipeline_v2/RESULT.json`。

- 已逐步重放旧发现实验的 512 条真实 4B 分叉；没有新增模型调用。
- 旧七状态三教师标签仍为 7/7 同向，独特行动方向为 0。
- 新发现池四个状态的采集策略各块仍无确认 CCE 偏好；参考块未完成。
- 输出判决 `NO_CCE_TRAINING_INCREMENT_NOT_ESTABLISHED`，训练标签文件为空。
- 不充分项分别记录：参考续行缺失、双方条件能力未认证、完整搜索/额外校验费用缺失、未建立非冗余监督。不将这批失败推广为 CCE 在所有任务无效。

## 使用

```powershell
python tools/kci_supervision_replay.py --study artifacts/20261122_cce_discovery_repair_v2/final_snapshot/study --output artifacts/一个全新的输出目录
```

新实验仍通过 discovery 的 `freeze -> collect -> fork -> analyze`。冻结时新增 `--capability-evaluations <collector_EVAL.json> <reference_EVAL.json>`，文件来自已有执行能力校准且须匹配相应模型；`--cost-ledger <json>` 记录 `witness_search` 和 `validation` 两项额外成本，每项含 `candidate_forwards/input_tokens/native_transitions/wall_seconds`。其中 validation 仅计额外原生核验，发现/验证分叉的采样已在各策略成本内，不重复相加。旧模型加载、冻结路径及模型占用检查保持原行为。

历史冻结目录不可原地重跑新版本；重分析单独输出，新增实验使用新目录。未访问封存测试内容，未更改其他 AI 的 PiSAs/KC8 开发文件，保持 main。

## 验证

本轮新增 13 个测试用例，连同 discovery 测试共 23 项通过：包含有效新信号的正向路径、新观察但终局同向的拒绝、参考缺失、确认方向冲突、UNKNOWN、费用缺失/非有限值、能力失败、checkpoint 不匹配、重复状态及旧训练入口提前退出。正向测试使用人工测试夹具，只验证代码可放行，不是新方法证据。

全套回归在 `code` 目录运行：**501 passed、2 skipped、395 subtests passed，30.60 秒**。从仓库根首次调用因 Python 找不到 `kci` 包而在收集时失败；保留该日志，切换正确工作目录后通过。测试与运行输出在正式结果目录，未以忽略失败的方式完成验收。
