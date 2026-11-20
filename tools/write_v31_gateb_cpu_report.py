r"""Source-checked Chinese report of actual small CPU learning and frozen LLM.

Reads exact saved experiment files and publishes every claim with its scope.
No inferred fake GPU cost or successful Qwen training claim.
"""
from pathlib import Path
from hashlib import sha256
import json,statistics
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261120_v31"
OUT=BASE/"KCI_V31_CPU_GATE_B_SMALL_SCALE_REPORT"
if OUT.exists():raise FileExistsError(OUT)
def j(path):return json.loads(path.read_text(encoding="utf-8"))
q=j(BASE/"FROZEN_QWEN25_V31_PRELFIGHT_01"/"SUMMARY.json")
qau=j(BASE/"FROZEN_QWEN25_V31_PRELFIGHT_01"/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json")
ne=j(BASE/"CPU_GATE_B_NEURAL_01"/"RESULTS.json")
per=j(BASE/"CPU_GATE_B_NEURAL_REPRODUCIBLE_03"/"AUDIT.json")
bandit=j(BASE/"CPU_GATE_B_TERMINAL_BANDIT_800_EPOCHS"/"RESULTS.json")
data=j(BASE/"V31_GATEA_FINAL_CANDIDATE"/"VALIDATION.json")
assert ne["status"]=="REAL_SMALL_TORCH_CPU_PARAMETER_TRAINING_COMPLETED"
assert per["status"]=="REAL_CPU_TRAINED_MODELS_WITH_NAME_PERMUTATION_AUDIT"
assert bandit["status"]=="REPRODUCED_LONG_BUDGET_TERMINAL_BANDIT_TRAINING"
assert qau["status"]=="INDEPENDENT_FROZEN_NAMED_ACTION_NATIVE_REPLAY_PASS"
assert data["case_mechanisms"]==63
assert ne["terminal_native_rollouts_created"]["reward_agrees_with_optimal"]==126
seeds=(7,1234,2025)
def test_row(variant):
    x=ne["means_and_sample_SD"][variant]["exploratory_test"]
    acc=x["accuracy"]
    sd=x["std_accuracy_sample"]
    flip=x["paired_flip_accuracy"]
    reg=x["mean_native_regret_vs_Qstar"]
    return f"| {variant} | {acc*100:.1f}% ± {sd*100:.1f}% | {flip*18:.1f}/18 | {reg:.4f} |"
def test_row_long():
    mr=[m["metrics"]["exploratory_test"] for m in bandit["methods"]]
    return "| terminal_bandit 800 epochs | "+f"{100*statistics.mean(x['accuracy'] for x in mr):.1f}% | {statistics.mean(x['paired_flip_correct'] for x in mr):.1f}/18 | {statistics.mean(x['mean_native_regret_vs_Qstar'] for x in mr):.4f} |"
count_new={
    "permutation_number":9,
    "test_case_count":36,
    "methods":4,
    "seeds":3,
    "tiny_trainable_horn_parameters":4,
}
model_weights=[]
for rec in per["records"]:
    ckpt=Path(rec["checkpoint_file"])
    assert ckpt.exists() and sha256(ckpt.read_bytes()).hexdigest()==rec["model_checkpoint_sha256"]
    model_weights.append(str(ckpt))
assert len(model_weights)==12
strong_robust=[x for x in per["records"] if x["variant"]=="neural_rule_value_pair"]
assert len(strong_robust)==3
assert all(r["untrained_test"]["accuracy"]==0.5
           and r["trained_metrics"]["exploratory_test"]["accuracy"]==1.0
           and r["worst_permutation_test_accuracy"]==1.0 for r in strong_robust)
assert all(x["metrics"]["exploratory_test"]["accuracy"]==1.0 for x in bandit["methods"])
lines=[
"# KCI V3.1 · Gate B 小规模 CPU 训练与冻结 LLM 行动验收",
"",
"**实验性质：真实 CPU 神经网络参数训练 + 实际原生终局策略奖励 + 冻结本地 Qwen2.5-7B 行动调用。它不是 Qwen3-4B RL，也不是完整多 Agent 策略训练。**",
"",
"## 一、问题与对照协议",
"",
"- 源数据：V3.1 经独立 Gate A 校验的63个合成机制、7类知识超图、3类业务流程。每个机制恰选一对相关与同主体安全历史，得到**类平衡126个早期决策状态**。",
"- 按完整知识图家族分组：训练4类图72状态，开发1类图18状态，探索性测试2类图36状态。开发/测试的Q-star用于事后评分，**未进入参数更新**。探索性测试图已参与数据开发，不是真盲测。",
"- 统一策略目标：同样的当前业务条件和权限，只改变 R 之前收到的事实，使 OPEN/SKIP 的长期原生Q-star排序逆转。规则、任务必需事实和历史由真实允许的actor观察提供。",
"- PyTorch CPU 版三种固定随机种子7/1234/2025；同模型结构160轮更新，无GPU或付费API。对比通用 MLP、带Horn规则消息传递归纳偏置的小网络、终局采样REINFORCE以及不看历史消融。",
"- 即时监督不人为制造假标签：在 preflight 状态 OPEN和SKIP均立即合法，因此仅靠立即违规标签没有区分信号；类平衡准确率在固定选择下为50%。",
"- 回报来自相同原生模拟器：共重放126个状态的双臂实际native动作（252完整终局结果）；该固定后续策略恰与当前Q-star两臂回报一致（126/126）。因此两种有监督信号**信息上可能等价**，实验只比较成对监督与单次采样的优化效率。",
"",
"## 二、真正训练的模型是什么",
"",
"**不要混淆四个可训练标量与通用LLM**：NeuralRule网络使用已给定的6个源节点、Horn超边和中间结论节点，将规则前提进行可微soft-AND和沿后继节点的soft-OR传递5层，最后通过四个可训练标量控制组合灵敏度与决策边界。",
"这是一种**几乎显式实现了规则推导的结构化神经模块**。它能学会的主要是规则传播的阈值标定，而不是从自然语言中自己发现规则体系。",
"MLP获得同样的允许输入事实和规则关联矩阵，但无图结构共享推导归纳偏置。输入解析器读取限定格式并转成特征；没有训练开放域自然语言解析。",
"长期价值监督是**带Q-star幅度权重的二分类BCE**，不是标准DPO训练，也不是Qwen神经策略的在线PPO/GRPO。",
"终局奖励方法为**on-policy sampled-action contextual bandit REINFORCE**：每次更新只观察选择动作的一条真实native终局奖励；固定后续执行策略遵循下一步即时CI保护。虽然为可重复验证预先缓存了双臂真实轨迹，优化器从未看到未选动作的回报。这不是完整的多角色RL。",
"",
"## 三、关键结果：未见知识图的36条平衡状态",
"",
"| 训练方式 | 最优早期动作准确率（3 seeds） | 成对历史Flip正确 | 平均相对Q-star regret |",
"|---|---:|---:|---:|",
test_row("mlp_value_pair"),
test_row("neural_rule_terminal_bandit"),
test_row("neural_rule_value_pair"),
test_row("neural_rule_value_no_history"),
test_row_long(),
"",
"一组重要额外控制：三个seed的规则神经网络**未训练前**在探索性测试为18/36=50%，训练长期配对Q-star监督后均为36/36=100%。相同训练后，针对普通事实S01..S06进行9种全体一致重命名，并颠倒收到的历史、规则子句顺序后，三个seed均仍为36/36。",
"通用MLP的测试准确率在不同随机seed和事实重命名下明显波动，说明它更依赖原事实位置而不是掌握了可共享的规则递归。",
"",
"### 终局奖励不是无效",
"",
"在160轮更新时，终局采样反馈的策略在未见图上平均52.8%，大量选择代价更高但安全的OPEN；这属于当前训练预算下的高方差、保守局部解。",
"将同一个终局回报采样实验单独延长到**800轮（5倍更新）**，三个seed全部到36/36（100%）且18/18组Flip正确。说明**长期配对监督更省优化更新，终局奖励可以在更多更新下同样学会**。这个差异不是数据上逻辑能力不可学习的证明，也不是GPU算力效率比较，因为生成精确Q-star本身需要大量CPU状态搜索。",
"",
"## 四、冻结本地语言模型的真实预备行动",
"",
"当前可调用的唯一冻结大型语言模型为本地Ollama 4-bit Qwen2.5-7B-instruct（**不是原Qwen3-4B**）。",
"为了消除动作索引位置偏差，统一要求模型输出完整OPEN/SKIP/ABORT行动名称的结构化JSON；温度0。",
"Zero-shot与提供4个**来自训练知识图、未见图不参加示例**的few-shot演示比较，在完全相同的18条开发与12条探索性测试状态上真实调用60次。",
"",
"| Qwen2.5-7B（权重未改） | 开发集动作准确率 | 未见图小测试准确率 | 未见图成对Flip |",
"|---|---:|---:|---:|",
]
for mode,name in (("zero_shot","zero-shot"),("four_train_examples","four training shots")):
    m=q["metrics"][mode]
    dv=m["dev"];te=m["exploratory_test"]
    lines.append(f"| {name} | {dv['correct']}/{dv['n']} | {te['correct']}/{te['n']} | {te['paired_flip_correct']}/{te['paired_total']} |")
lines+= [
"",
"这些动作已由独立脚本重新加载对应的真实native状态，执行保存的模型选择并校验最终业务数据库状态与接收者账本，60/60回放通过。模型未训练，few-shot变化很小且小样本方差高，不能据此宣称实际LLM已有可靠长期知识条件化能力。",
"",
"## 五、统计/科学限制",
"",
"1. V3.1只包括受控虚构业务、有限Horn语法和公开推理规则；不是跨真实领域的自由语言CI对齐。",
"2. 主指标36个探索性测试状态来自2个已用于开发的知识图家族。seed SD不是跨独立业务任务的统计置信区间；测试结果不能写作大规模泛化结论。",
"3. 用关系规则消息传递获得100%很大一部分来自**人工设计的正确结构归纳偏置**，非通用模型通过长期奖励自动学会复杂推理。显式符号规划器同样100%，必须保留。",
"4. 长期Q-star监督训练需要正确的完整模型可见事实提取和显式规则，且Q-star是原生模拟器最优续行，不等于模型Q-pi。即便这个CPU代理成功，目标4B的实际off-policy优化或RL梯度仍未验证。",
"5. 终局采样的800轮5倍反馈预算有利于收敛，这一比较展示样本/更新效率的候选方向，尚非相同总算力/环境回放次数下的严格匹配实验。",
"6. 当前只训练Coordinator preflight动作。数据虽有多个角色的native oracle示范，仍未训练一个自主多角色的LLM Agent；对于论文的多Agent完整策略学习，Gate B尚未通过。",
"7. Qwen2.5只做冻结无权重更新的ICL，不能当成目标Qwen3-4B的微调实验，且四shot提示长度不同于零shot。",
"",
"## 六、研究建议",
"",
"**可确认：** 数据包含有用的长期历史条件化信息；训练一个强结构归纳偏置的小策略可以利用精确配对Q-star监督，并对未见知识图里的标准事实重新编号保持不变；单次终局反馈在该优化设置下比低方差成对信号慢。",
"**尚未确认：** 在通用LLM上学习出可泛化的知识积累与决策；在没有显式规则图/解析器时学习到相同能力；相比精确可见规则规划器产生增益。",
"**下一关键动作：** 获取能真正训练的小型预训练LM（或恢复目标Qwen3-4B GPU），固定同模型、相同数据、动作协议、计算与样本预算；比较真实SFT/terminal-RL/value-supervised学习，以及整个Agent多角色终局完成与泄漏。没有必要继续对当前模板无限扩增。",
"",
"## 七、可复现文件",
"",
"- CPU_GATE_B_NEURAL_01/RESULTS.json：12次160轮训练的全部seed/拆分/动作准确率、Flip、regret与源设置。",
"- CPU_GATE_B_NEURAL_REPRODUCIBLE_03/AUDIT.json + MODEL_WEIGHTS/*.pt：12个可重载的小型PyTorch模型参数及9种事实重编码鲁棒性审计。",
"- CPU_GATE_B_TERMINAL_BANDIT_800_EPOCHS/RESULTS.json：800轮采样终局奖励实验真实训练结果。",
"- FROZEN_QWEN25_V31_PRELFIGHT_01/CONFIG、MODEL_ACTUAL_DECISIONS、SUMMARY、INDEPENDENT_NATIVE_REPLAY_AUDIT：模型名字与digest、60次真实行动、终局回放。",
"- tools/train_v31_cpu_policy_probe.py、tools/audit_cpu_policy_gateb_permutations.py、tools/audit_terminal_bandit_long_budget_v31.py：本实验训练和复制脚本。",
"- tools/evaluate_v31_frozen_ollama_fewshot.py、tools/audit_v31_frozen_qwen_icl.py：冻结Qwen JSON动作与原生环境状态回放。",
]
OUT.mkdir()
(OUT/"GATE_B_CPU_SMALL_SCALE_CN.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
audit={
    "status":"SOURCE_VERIFIED_GATE_B_CPU_PRELIMINARY",
    "train_graphs":4,"dev_graphs":1,"exploratory_graphs":2,
    "seeds":[7,1234,2025],
    "training_worlds":72,"dev_worlds":18,"test_worlds":36,
    "neural_rule_param_count":4,
    "first_phase_updates":160,
    "terminal_long_updates":800,
    "permutations_renaming_test_count":9,
    "untrained_rule_model_test_accuracy":0.5,
    "value_rule_test_accuracy":ne["means_and_sample_SD"]["neural_rule_value_pair"]["exploratory_test"]["accuracy"],
    "value_rule_all_permuted_test_accuracy":min(x["worst_permutation_test_accuracy"] for x in strong_robust),
    "terminal_bandit_160_test_accuracy":ne["means_and_sample_SD"]["neural_rule_terminal_bandit"]["exploratory_test"]["accuracy"],
    "terminal_bandit_800_test_accuracy":statistics.mean(x["metrics"]["exploratory_test"]["accuracy"] for x in bandit["methods"]),
    "plain_MLP_test_accuracy":ne["means_and_sample_SD"]["mlp_value_pair"]["exploratory_test"]["accuracy"],
    "no_history_rule_test_accuracy":ne["means_and_sample_SD"]["neural_rule_value_no_history"]["exploratory_test"]["accuracy"],
    "frozen_qwen_zero_test_accuracy":q["metrics"]["zero_shot"]["exploratory_test"]["accuracy"],
    "frozen_qwen_four_shot_test_accuracy":q["metrics"]["four_train_examples"]["exploratory_test"]["accuracy"],
    "frozen_Qwen_inference_calls":q["total_model_actions"],
    "trained_original_Qwen3_4B":False,
    "trained_local_7B_Qwen":False,
    "tiny_CPU_neural_models_trained":12,
    "all_60_Qwen_actions_native_replayed":True,
    "terminal_arm_source_matches_exact_Qstar_for_fixed_suffix":True,
    "native_oracle_not_model_Qpi":True,
    "project_grade":"PRELIMINARY_CPU_SURROGATE_GATE_B_NOT_PAPER_OR_LLM_PASS",
    "supporting_source_file_sha256":{
        str(p.relative_to(BASE)):sha256(p.read_bytes()).hexdigest()
        for p in (
            BASE/"CPU_GATE_B_NEURAL_01"/"RESULTS.json",
            BASE/"CPU_GATE_B_NEURAL_REPRODUCIBLE_03"/"AUDIT.json",
            BASE/"CPU_GATE_B_TERMINAL_BANDIT_800_EPOCHS"/"RESULTS.json",
            BASE/"FROZEN_QWEN25_V31_PRELFIGHT_01"/"SUMMARY.json",
            BASE/"FROZEN_QWEN25_V31_PRELFIGHT_01"/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json"
        )},
}
(OUT/"AUDIT_SUMMARY.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
