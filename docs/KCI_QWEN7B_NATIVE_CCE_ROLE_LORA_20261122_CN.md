# KCI-Align：真实 7B Agent 执行、同状态 CCE 与角色 LoRA 训练闭环（2026-11-22）

**目的**：从原先的结构化 CPU 机制实验，实际进入预训练语言模型在动态 CI 工作流上的行为与方法数据。保留原本的 recipient-knowledge / long-horizon safe value / downstream responsibility 三个研究 Insight。所有新评测使用现有原生 A→R→B→C 验证器与自然语言可见观察，不重新制造任务与标签。

## 1. 真正使用了什么模型

本机 Ollama 已安装 qwen2.5:7b-instruct-q4_K_M（约 4.7GB 量化权重），在 Windows CPU 上正常执行。单次上游决策输入约数百 token，首次模型调用用时约10秒，后续几秒。已配置 OPENAI_API_KEY，但实际 OpenAI API 小型连通性请求在30秒超时；本轮不依赖外部付费 API。nvidia-smi 没有报告可用 CUDA 设备。此项目机器拥有的 Ollama GGUF 量化模型仅支持本轮推理；目前未更新真实 7B 权重。

### 原生交互合同

- 语言模型实际看到：tools/kci_llm_text_surface.py 将当前角色获准观察的任务、规范、真实已送达 receipt、公开推断规则、证据包及成本合同转为自然语言。
- 模型只输出一个动作选项字母。映射到原生 action，实际读取、消息交付、Recipient R 知识更新、源 Agent 及业务提交全由既有 native env 真实执行。
- A_SEND、B/preflight、source/release 和最终业务COMMIT/ABORT是真实预训练模型决策；非争议的源读取、业务状态检查、固定业务路线由所有方法共用的 native scripted adapter 执行。
- 终局 CI、任务完成和 reward 都不是模型自评。没有把 hidden Qstar、native state 或未来 reward 输入 prompt。
- 记录每个模型提示的 SHA256、行动、native before/after 哈希、实际送达消息数、token 数和动作审计。

## 2. 真实预训练 Qwen7B 的单步学习缺口

在全部已有28派生世界，对 RecordsAgent 的两个当前合法来源包直接使用预训练 Qwen7B 推理：
- 28/28 模型输出均为有效 native 选项；
- 与原生理想长期值比较，24个严格非平局上游选择中 16个正确（train 9/14，dev 3/4，developer-seen test 4/6）；
- 按业务图根，有明显的包名交换一致性/作用差异；不能把24个强相关状态说成24个独立场景。
- 此处仅诊断 A 行动排序，不把 Qstar 标签用于实际模型提示或训练。

## 3. 真实 7B 完整 Agent 路径：第一个可直接训练的失败现象

在 developer-seen TEST 的两个图根 DG18/DG19、各4个成本/包名条件，完整执行8条轨迹。
- 默认模型 8/8 在 B/preflight 选择 SKIP_PRIVATE_CHANNEL；
- 随后来源角色在 release 选择 ABORT，而不是完成必要来源披露；
- 终局安全完成 0/8，最终 CI 违规0/8，任务未完成8/8，reward全为0。
- 独立测试一个泛化的“考虑未来不可逆决策/不要不必要放弃”自然语言规划提示，结果依然 0/8；不应夸大普通提示能解决该问题。

这是原始方法假设的直接可执行例：局部合法且便宜的行动导致后续可用的合规任务路径失败。模型懂得避免即时泄漏，但没有正确预估下游。

## 4. 同一真实语言模型的 B 同状态反事实

对上述8条真实完整轨迹的 B/preflight 原生状态进行独立哈希重放。分别强制其两个当前均合法的 B 动作，后续再次交给冻结的 Qwen7B，其他 native 工具逻辑保持不变。

- OPEN_PRIVATE_CHANNEL：8/8 完整合规业务成功。来源 Agent 主动选 VERIFY_PRIVATE_R，随后 BusinessExecutor 自行选 COMMIT_RESULT。
- SKIP_PRIVATE_CHANNEL：0/8 成功；后续模型选择 ABORT，终局CI违规0/8。
- 同状态偏好因此均是 OPEN > SKIP。实际成功 reward 根据业务条件不同：DG18 matched 0.609；DG18 tradeoff 0.517；DG19 matched 0.608；DG19 tradeoff部分0.496/0.608。
- 16条真实冻结模型 suffix；两个实际图根×四个派生世界。它们是开发者已见的 TEST 诊断，不作为 TRAIN 偏好来源。
- 单条确定性后续模型执行不是对高随机性模型 Qpi 的可靠统计估计；目前只证明这种行为在固定的当前策略/解码设置下可复现。

这个结果直接确认：长期 CCE 的 B 局部信用并不需要理想规划器；相同真实模型在两个合法早期动作下可以分别表现出安全成功/安全失败。

## 5. TRAIN-only 真实语言模型 CCE 数据

使用独立的 A-only TRAIN 模型选择，按 native 工具重放抵达 TRAIN 的 B/preflight；挑选4个训练图根 DG01/DG05/DG09/DG10、每根 matched/tradeoff 两个世界，共8个B同状态分叉。

真实 Qwen7B 在原始 TRAIN 的8/8个 B状态均选择 SKIP；但强制 OPEN后的冻结模型后续8/8安全成功（reward 0.202–0.748），强制 SKIP后8/8安全放弃，因此生成：
- 8对来自实际冻结预训练模型完整续行的 Coordinator OPEN > SKIP DPO 偏好；
- 4个独立训练图根，16次原生完整后续执行；
- 每条DPO记录包含：完整授权自然语言prompt、原native动作列表、chosen/rejected动作与选项字母、真实native状态哈希、真实终局CI/任务和reward；
- 没有任何开发者已见TEST数据或理想Qstar动作充当TRAIN教师。

这些记录是**bootstrap**，不是最终证明学习了知识条件化策略：8对全是 OPEN > SKIP，模型如果只记住“总是OPEN”也能拟合。放大训练必须先修复 SourceAgent 能力并重新冻结当前团队采样B偏好，保留确实应该SKIP的合法任务正例，否则方法退化成保守策略。这个检验直接对应之前的DPR和能力条件化Qpi，不属于新方向。

## 6. 合法与非法披露的角色训练数据

现有16个TRAIN世界中的真实native source/release决策可复核生成：
- 28条 RAW 会即时造成 CI 违规、VERIFY合法的局部下游纠错偏好；
- 72条双方当前合法、且在已有冻结CPU策略的原生续行核查下 RAW 安全、任务不劣而成本更低的合法披露正例；
- 总100条SourceAgent角色训练偏好，4个TRAIN图根。合法 RAW 正例的证据来自本地冻结CPU actor/native验证，明确**不是来自本次Qwen7B策略估计**；
- 每条偏好仍是模型可见的自然语言角色行动选项，不直接给出结构化隐藏oracle标签。

这使下游LoRA训练不只学“总选VERIFY/拒绝”，有机会学习依接收者知识确定是否允许 RAW。

## 7. 真实 LoRA-DPO 的可执行桥接

代码：tools/kci_role_lora_dpo.py 接受上述 TRAIN-only JSONL，检查角色、执行选项、letter映射、state hash、evidence来源，使用项目已有HFActionPolicy和 PEFT新LoRA参数计算 action-token DPO：
- chosen/rejected真实选项字母概率；
- 冻结基座作为参考，标准DPO log-sigmoid差分；
- 所有梯度仅作用于role adapter训练参数；保存独立adapter/checkpoint和训练信息。
- 在本机仅进行CPU可微分数学自检，25次更新中 OPEN-SKIP 的 log-odds 从0增至约2.540，尾部loss约0.0067；**不是7B真正的LoRA参数更新**。
- tools/kci_role_adapter_router.py 提供一个共享HF底座、角色条件化PEFT adapter切换：RecordsAgent/A、Coordinator/B、SourceAgent/R（DirectoryAgent等来源角色共享R adapter），其余业务工具角色保留基座。尚未在可训练4B设备上验证此runtime。

本机无GPU、仅有Ollama量化模型与无PEFT Python依赖，无法对当前GGUF直接执行QLoRA训练。原始项目已经有GPU版本HFPolicy可直接复用；在有CUDA和本地HF模型权重的环境，上述训练入口可以用于小规模B/R更新。由于B偏好目前全OPEN，强烈建议先按R local repair → freeze R and resample B → B training → freeze team and collect A CCE的顺序执行，而不是直接把当前8条B样本当论文最终训练集。

## 8. 测试、成果与下一次实验

- 真实Qwen7B A decision：28个；
- Qwen7B完整原生TEST执行：默认8个、一般规划提示8个；
- 开发TEST B两动作冻结续行：16条；
- TRAIN 4图根B两动作冻结续行：16条；
- TRAIN B原模型实际OPEN/SKIP选择：8个，8/8 SKIP；
- TRAIN SourceAgent数据100对、TRAIN Coordinator数据8对；
- CPU DPO真实偏好记录数学验证通过，相关pytest 15 passed；
- 本机目前没有新的预训练大模型参数训练，不能据此写7B对齐已有效。

下一步直接推进的是：R角色LoRA微调，并在相同模型上重新采B续行，检查是否出现知识状态条件化的OPEN/SKIP双向偏好，最终进行上游A训练；重点评价任务完成、CI风险和真正未见任务根泛化。不要再扩展S01-S06的表面编号以冒充独立新机制。
