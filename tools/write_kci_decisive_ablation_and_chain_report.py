"""Compile a transparent source-checked Chinese report on P0 and A/B/C pilot."""
from pathlib import Path
from collections import Counter
from hashlib import sha256
import json
R=Path(__file__).resolve().parents[1]
BASE=R/"artifacts"/"20261120_v31"
P0=BASE/"P0_LABEL_VS_QMARGIN_BCE_20261120"
LOW=BASE/"P0_LOW_DATA_LABEL_VS_QMARGIN_ABLATION"
CHAIN=BASE/"DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01"
DOC=R/"docs"/"KCI_P0_LABEL_VS_QSTAR_AND_DYNAMIC_AGENT_CHAIN_CN.md"
if DOC.exists():raise FileExistsError(DOC)
def j(p):return json.loads(p.read_text(encoding="utf-8"))
a=j(P0/"ABLATION_RESULTS.json")
b=j(LOW/"RESULTS.json")
c=j(CHAIN/"VALIDATION.json")
d=j(CHAIN/"INDEPENDENT_CAUSAL_REPLAY_AUDIT.json")
assert a["status"]=="ACTUAL_MATCHED_CPU_ABLATION_COMPLETED"
assert b["status"]=="LOW_DATA_PAIRED_ACTION_LABEL_VS_QMARGIN_COMPLETED"
assert d["status"]=="INDEPENDENT_DYNAMIC_CHAIN_CAUSAL_ACTION_REPLAY_PASS"
cases=[json.loads(line) for line in
       (CHAIN/"CHAIN_CAUSAL_PAIRS_EVAL_ONLY.jsonl").read_text(encoding="utf-8").splitlines()
       if line.strip()]
counts=Counter()
for entry in cases:
    q=entry["upstream_A_values_given_exact_B_continuation"]
    risk=q[entry["related_package"]]
    safe=q[entry["benign_package"]]
    key="risk" if risk>safe else "safe" if risk<safe else "tie"
    counts[entry["source_mode"],key]+=1
assert counts["matched","safe"]==14 and counts["tradeoff","risk"]==10 and counts["tradeoff","tie"]==4
text=[
"# KCI: 决定性消融与动态 A→R→B→C 原生验证",
"",
"结论级别：已验证 CPU 合成机制与训练对照；不是 Qwen3-4B 的 RL 结果，V3.1 冻结数据未修改。",
"",
"## P0：最优动作标签与 Q-star 数值幅度是否不同？",
"",
"同一四参数 NeuralRule、72个训练状态、相同训练图分割、相同Adam及学习率0.03，五个完全配对随机种子，每个方法160次全批更新。三种方法只改变同一个正确标签的BCE权重：统一1、真实Q-star幅度加权、或打乱幅度加权。三者均在全部五种子达到未见图36/36、历史Flip18/18、Regret=0。单纯Q-star数值幅度没有给出独立的最终性能收益。原有的数值加权BCE不是CCE原始算法，也不是DPO。",
"",
"| 方法 | 未见图准确率 | 历史Flip | Regret |",
"|---|---:|---:|---:|",
]
for method,title in (
    ("label_BCE","普通最优动作标签BCE"),
    ("Qmargin_weighted_BCE","真实Q-star幅度加权BCE"),
    ("shuffled_Qmargin_BCE","打乱Q-star幅度BCE")):
    stat=a["aggregate"][method]["exploratory_test"]
    text.append(f"| {title} | {stat['mean_accuracy']:.1%} | {18*stat['mean_flip_accuracy']:.0f}/18 | {stat['mean_regret']:.4f} |")
text += [
"",
"前述结论不能粗暴写成数值幅度没有任何作用。使用每次4/8/16/36个训练机制、相同五种子和预定20/40/80/160检查点时，真实权重方法在第40步的成本敏感Regret显著低于两对照，虽无稳定动作准确率提升。第80步普通BCE经常反而占优，第160步全部回报和Regret相同。这是早期风险优先级的局部实验现象，需要在严格预先固定预算和其他模型上复验。",
"",
"| 训练机制数 | 第40步普通BCE Regret | 真实Q差权重 | 打乱Q差权重 |",
"|---:|---:|---:|---:|",
]
for n in ("4","8","16","36"):
    rows=b["aggregate"][n]["40"]
    text.append(f"| {n} | {rows['label_BCE']['mean_test_regret']:.4f} | {rows['Qstar_weighted_BCE']['mean_test_regret']:.4f} | {rows['shuffled_Qstar_BCE']['mean_test_regret']:.4f} |")
text += [
"",
"必须同时计入获得精确Q-star标签及其幅度的真实原生搜索成本。给优化器相同更新次数，不等于拥有相同总监督成本。",
"",
"## P1：从预置Archive历史变成上游真实披露",
"",
"另建一套独立的合成执行器而非修改V3.1。RecordsAgent先实际调用读取两个被认证的模拟工具证据包，再从同一个合法状态选择实际发送；事实消息真正到达R后才更新R知识。然后Coordinator实际检查业务并选择OPEN/SKIP。后续来源Agent可决定直接发送原文或者使用私下验证，BusinessExecutor在检查完成后实际提交日历、订单或审阅任务结果。",
"",
"| 验收 | 独立执行与回放 |",
"|---|---:|",
f"| 图机制 | {c['mechanism_roots']} |",
f"| A当前同状态两合法证据选项 | {d['counts']['same_A_source_state_two_legal_actions']} |",
f"| B仅接收历史不同而严格最优动作Flip | {d['counts']['B_history_only_strict_flips']} |",
f"| 真正的A/B/C终局续行 | {d['counts']['B_action_native_Qstar_arms_rechecked']} |",
f"| 独立重放B/C行动转移 | {d['counts']['native_action_transitions_replayed']} |",
f"| 原生业务完成提交 | {d['counts']['real_C_business_commits_replayed']} |",
f"| B下游当前真实原文违规/私下验证纠正 | {d['actual_downstream_same_state_leak_vs_correction_pairs']} |",
"",
"这28次下游纠正实验在同一后续状态比较RAW与VERIFY，真正产生违规的是后续原文披露；上游A此前发送的证据在发送时合法。不能把下游Agent的即时违规当成A的即时违规。B的未来最优准备行为依赖R实际新收到的记录，不是模拟器预先写好的Archive历史。",
"",
"上游也存在真实延迟价值取舍：当两证据包成本相同，14/14次优选不会迫使下游付出额外保护步骤的证据包；当安全证据包额外需要4步办理，10/14次反而优选当前合法但需要下游保护的快证据包，4/14种两路线价值持平。任务机制不再强制上游永远隐藏信息。",
"",
"## 当前结论和不可越界的范围",
"",
"1. V3.1的Horn规则近乎被写进四参数模型前向传播；普通最优动作标签足以达到相同最终性能，不能由当前数值加权BCE声称独立CCE贡献。",
"2. 早期成本敏感Regret的改善值得保留为假说，但训练与构造精确Q-star的开销未进行公平成本折算。",
"3. 新A→B→C链使用确实执行的原生工具、送达记录与下游补救，而非预填评价标签；但执行策略是确定性参考脚本与原生枚举，并没有训练共享LLM、没有Qpi反馈，也不属于完整的CCE/DPR训练结果。",
"4. 该结构仍然采用受控标准事实、显式推理规则、虚构组织隐私规范，显式规则规划器仍能达到最优。小规模机制的可执行性不等于论文级复杂自然语言任务验证。",
"5. 下次最关键的实验必须在同一可训练模型及相同业务、观测、动作格式下，于A合法信息披露和B可纠正不当披露的两处分别比较：Action SFT、Terminal RL、真实反事实信用分配，以及不含DPR局部纠正的消融。还要报告额外采样/原生续行开销。",
"",
"## 可复现材料",
"",
"scripts: tools/decisive_v31_label_vs_qmargin_ablation.py; tools/ablate_v31_low_data_q_weights.py; code/kci/upstream_downstream_chain_pilot.py; tools/build_kci_dynamic_chain_pilot.py; tools/audit_kci_dynamic_chain_pilot.py; code/tests/test_kci_dynamic_chain_pilot.py。",
"",
"evidence: artifacts/20261120_v31/P0_LABEL_VS_QMARGIN_BCE_20261120/ABLATION_RESULTS.json; artifacts/20261120_v31/P0_LOW_DATA_LABEL_VS_QMARGIN_ABLATION/RESULTS.json; artifacts/20261120_v31/DYNAMIC_UPSTREAM_DOWNSTREAM_CHAIN_PILOT_01/。",
]
DOC.write_text("\n".join(text)+"\n",encoding="utf-8",newline="\n")
audit={
    "status":"SOURCE_CHECKED_P0_ABLATION_AND_DYNAMIC_CHAIN",
    "fixed_original_v31":True,
    "five_seeds":a["fixed_seed_set"],
    "all_three_BCE_methods_test_accuracy":{name:a["aggregate"][name]["exploratory_test"]["mean_accuracy"] for name in a["methods"]},
    "P1_dynamic_native_paths":d["counts"]["B_action_native_Qstar_arms_rechecked"],
    "P1_history_only_B_flips":d["counts"]["B_history_only_strict_flips"],
    "P1_downstream_native_correction_cases":d["actual_downstream_same_state_leak_vs_correction_pairs"],
    "no_trained_shared_LLM":True,
    "is_paper_method_contribution":False,
    "source_sha256":{str(p.relative_to(R)):sha256(p.read_bytes()).hexdigest()
       for p in (
            P0/"ABLATION_RESULTS.json",
            LOW/"RESULTS.json",
            CHAIN/"INDEPENDENT_CAUSAL_REPLAY_AUDIT.json")},
}
DOC.with_suffix(".audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
