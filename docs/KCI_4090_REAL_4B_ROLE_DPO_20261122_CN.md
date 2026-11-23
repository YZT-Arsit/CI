# KCI-Align — RTX 4090 上真实 Qwen3-4B 角色 LoRA 与 CI 原生执行

实验日期：2026-11-22（服务器北京时间跨至11月23日）
模型：Qwen3-4B-Instruct-2507，NF4 4bit，bf16，LoRA rank8（q_proj、v_proj）。
设备：NVIDIA RTX 4090 D，24GB。训练实际使用显存约5.5–5.7GB。
所有训练与评估使用原生 A→R→B→C 模拟器的真实工具动作、接收者已送达记录、规范真值和业务回报。各角色共享同一个预训练底座；只对目标角色启用独立LoRA。

## 1. 真正的4B参数更新已完成

Coordinator/B：64个实际DPO优化步，lr=5e-5，beta=2，batch=1，8条TRAIN偏好，梯度非零，Adapter单独保存。B的8条训练偏好来自之前冻结Qwen2.5-7B的真实续行，是跨模型教师，只作为训练接口连通性验证。

SourceAgent/R：128个实际DPO优化步，lr=2e-5，beta=2，batch=2。训练集合：28条即时RAW违规→VERIFY合法纠正、72条合法有效RAW正例、104条原生成功任务路径相对于安全放弃ABORT的可验证见证，共204条偏好，4个TRAIN图根，40种不同角色可见文字提示。104条为可执行且已检验的任务成功见证，不是隐藏最优Qstar教师。

两份真实LoRA权重均通过PEFT保存，并从服务器备份回用户本地电脑，各约11.3MiB。没有重新下载原始4B或覆盖先前GPU实验。

## 2. B角色的真实负结果：它不是当前4B的失败位置

| 指标 | 原始Qwen3-4B | 只训练B-LoRA |
|---|---:|---:|
| TRAIN 选择OPEN | 16/16 | 16/16 |
| DEV 选择OPEN | 4/4 | 4/4 |
| 探索TEST选择OPEN | 8/8 | 8/8 |
| DEV合规完成 | 0/4 | 0/4 |
| 探索TEST合规完成 | 8/8 | 8/8 |
| DEV及探索TEST发生CI违规 | 0 | 0 |

原模型在28/28个B状态已选择OPEN。原始平均P(OPEN)：TRAIN 97.75%，DEV 96.04%，TEST 99.76%。B训练后均接近100%，但没有改善任务。DEV 失败实际发生在下游SourceAgent的release步骤，该角色直接ABORT。

## 3. SourceAgent/R角色的真实正结果

| 指标 | 原始4B | 只训练R-LoRA |
|---|---:|---:|
| DEV（DG14）合规完成 | 0/4 | 4/4 |
| 探索TEST（DG18、DG19）合规完成 | 8/8 | 8/8 |
| DEV和探索TEST最终CI违规 | 0 | 0 |
| DEV平均原生终局回报 | 0 | 0.505 |
| DEV首次来源决策P(VERIFY) | 35.02% | 98.50% |
| 探索TEST首次来源决策P(VERIFY) | 89.22% | 99.99% |

R-LoRA只在SourceAgent的release决策启用，其余A/B/C仍是完全相同的冻结原始4B。模型不再在DEV状态直接ABORT，而是进行合法私有验证，并由业务角色完成真实任务提交。

DEV四条仅来自一个未在TRAIN使用的知识图根DG14；探索TEST的两个图根已参与此前开发，因此不是论文级盲测。不过这是真实预训练4B权重更新后的完整Agent执行改进，不是CPU规则模型的替代指标。

## 4. 冻结R-LoRA后，重新计算4B本身的B反事实续行

在16个TRAIN世界，首先让冻结Qwen3-4B选上游A证据包；B同一状态分别强制两个当前合法行动OPEN/SKIP；剩余执行由相同冻结4B+R-LoRA完成。共32条真实原生后续轨迹，生成16条有效B偏好：

- 14条来自合法任务成功与失败差异；
- 2条来自真实CI违规与无违规差异；
- 16条全部指向OPEN。该结论不由Qwen2.5-7B教师或Qstar标签替代。

目前B的真实Qpi偏好仍然是单向OPEN，继续重复B-LoRA不构成有效的长期知识条件化训练增量。下一步应该直接研究当前已修复下游能力时A的合法来源包分叉，以及不同接收者历史如何改变B后续的可实现价值。

## 5. 独立审计与保存位置

Windows端使用独立原生执行器重新读取完整模型决策结果并重放：
- B训练前后比较：24条完整轨迹，272次原生状态转移。
- R训练前后比较：24条完整轨迹，292次原生状态转移。
- 共48条真实4B执行轨迹、564次操作。状态哈希、native verdict、R收到的新增消息、CI结果、任务完成及终局回报与保存的执行记录全部一致。
- GPU训练的偏好数据SHA256与TRAIN实际文件一致；训练的两个Adapter权重已备份并计算SHA256。
- 既有主分支、原始V3.1数据和旧GPU实验不受影响；SSH登录凭据没有进入实验文件。

服务器保存目录：
/root/kci_native_4090_20261122/artifacts/gpu_4090_B_CCE_Qwen3_4B_v1/adapter
/root/kci_native_4090_20261122/artifacts/gpu_4090_R_CI_task_Qwen3_4B_v1/adapter
/root/kci_native_4090_20261122/artifacts/gpu_4090_B_native_eval_v2
/root/kci_native_4090_20261122/artifacts/gpu_4090_R_native_eval_v1
/root/kci_native_4090_20261122/artifacts/gpu_4090_B_own_Qpi_after_R_v1

Windows保存目录：
D:\research\longhistory\artifacts\20261122_kci_4090_gpu_real_llm_v1

本轮对ICML研究主线的价值：验证了长期CI系统中对真实失败角色分配可信局部训练信号，能在4B上改变完整多Agent的安全任务完成。CCE算法在大模型上的独立长期收益仍需以新R能力下的A反事实和知识条件化双向偏好检验，而不需要重新寻找CI研究动机。
