"""Source-checked Chinese diagnosis of two implementation bugs and post-hoc repair.

This report reads immutable native action traces and compares original
vs isolated model-code repairs. The original V3.1 dataset is unchanged.
"""
from pathlib import Path
from hashlib import sha256
from statistics import mean
import json

R=Path(__file__).resolve().parents[1]
B=R/"artifacts"/"20261120_v31"
DOC=R/"docs"/"KCI_CHAIN_SHARED_ACTOR_REPAIR_DIAGNOSIS_CN.md"
META=R/"docs"/"KCI_CHAIN_SHARED_ACTOR_REPAIR_DIAGNOSIS_AUDIT.json"
if DOC.exists() or META.exists():raise FileExistsError("Refuse overwrite")
def j(p):return json.loads(p.read_text(encoding="utf-8"))
old=j(B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_01"/"RESULTS.json")
fix=j(B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01"/"RESULTS.json")
native=j(B/"CHAIN_NATIVE_ORACLE_BASELINES_01"/"SUMMARY.json")
two=j(B/"CHAIN_CPU_HORN_RULE_BY_ACTION_INTERACTION_2X2_01"/"RESULTS.json")
stress=j(B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"STRESS.json")
stress_audit=j(B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"INDEPENDENT_REPLAY_AUDIT.json")
diagnosis=j(B/"CHAIN_CPU_HELDOUT_FAILURE_DIAG_01"/"DIAGNOSIS.json")
dpr=j(B/"CHAIN_DPR_BACKTRACKED_ACTION_INTERACTION_REPAIRED_01"/"RESULTS.json")
fix_audit=j(B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01"/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json")
dpr_audit=j(B/"CHAIN_DPR_BACKTRACKED_ACTION_INTERACTION_REPAIRED_01"/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json")
assert old["status"]==fix["status"]=="CPU_SHARED_ROLE_POLICY_METHOD_COMPARISON_REAL_NATIVE_ROLLOUTS"
assert fix_audit["status"]=="INDEPENDENT_SHARED_CPU_ACTOR_NATIVE_REPLAY_PASS"
assert dpr_audit["status"]=="INDEPENDENT_BACKTRACKED_DPR_NATIVE_REPLAY_PASS"
assert stress_audit["status"]=="INDEPENDENT_SAVED_PYTORCH_ACTOR_STRESS_NATIVE_REPLAY_PASS"
assert native["status"]=="EXACT_NATIVE_PILOT_REFERENCE_CEILING_VERIFIED"
safes=[x for x in diagnosis["per_condition"]
       if x["mode"]=="action_sft" and not x["true_risk_from_public_rule"]]
false_alarms=sum(x["soft_future_risk"]>0.5 for x in safes)
assert (false_alarms,len(safes))==(28,28)
def arr(table,method,key,split="exploratory_test"):
    return [x["scores"][split][key] for x in
        table["per_seed_and_heldout_full_native_model_action_paths"]
        if x["mode"]==method]
method_names=[
    ("action_sft","Action SFT"),
    ("sft_counterfactual_coverage","Action SFT + 相同反事实状态"),
    ("terminal_reinforce","Terminal REINFORCE"),
    ("cce_reference_dpo_proxy","CCE参考Q-star成对DPO代理"),
    ("cce_reference_dpo_plus_dpr_proxy","CCE + 来源动作DPR代理"),
]
for name,_ in method_names:
    assert len(arr(fix,name,"safe_success"))==3
assert arr(fix,"action_sft","safe_success")==[8,8,8]
assert arr(fix,"cce_reference_dpo_proxy","safe_success")==[8,8,8]
factor=two["aggregate"]
expect={
    "ORIGINAL_soft_infer_no_interaction":[0,0,0],
    "FIXED_logic_only":[0,0,0],
    "INTERACTION_only_SOFT_INFER":[4,4,4],
    "BOTH_zero_preserving_and_action_interaction":[8,8,8],
}
assert all(factor[k]["exploratory_safe_success_seeds"]==v for k,v in expect.items())
oracles=[json.loads(s) for s in
    (B/"CHAIN_NATIVE_ORACLE_BASELINES_01"/"TRACES_EVAL_ONLY.jsonl").read_text(encoding="utf-8").splitlines()
    if s]
q_mean=mean(x["actual_native_Qstar"] for x in oracles
            if x["split"]=="exploratory_test")
sft_return=fix["aggregate"]["action_sft"]["exploratory_test"]["mean_native_return"]["mean"]
repairs=[x["DPR_actual_failure_verified_unique_pairs"] for x in dpr["per_seed_optimized_models"]]
assert repairs==[3,3,3]

lines=[
"# KCI A-R-B-C 共享CPU策略：失败定位与修复记录",
"",
"研究结论等级：**可执行的合成CPU环境与小型神经策略工程修复**。不是Qwen模型微调，不是KCI-Align的CCE/DPR算法收益证明，也不是正式盲测。",
"",
"## 1. 两个独立定位的实现缺陷",
"",
"**缺陷一：错误软逻辑传播。** 旧代码对Horn规则使用sigmoid(k × (匹配前提数-全部前提数+0.5))。缺少前提也可能产生正激活，反复传播使原本不合法的推导边被错误触发。",
f"在{len(safes)}个真正安全的B同前缀分支里，旧模型的未来风险值有{false_alarms}/{len(safes)}超过0.5，错误地把无风险状态当成风险状态。修复采用零保持的AND/OR t-norm；来源合法可见的0/1证据下，缺少任何前提都不得推导结论。",
"",
"**缺陷二：风险与候选动作脱钩。** 旧版B的OPEN/SKIP动作共享同一未来风险标量，仅靠动作one-hot让小MLP从少量样本拟合高阶交互。修复加入五个初始权重均为0的可训练风险×候选动作基函数，分别作用于A证据包、B的OPEN/SKIP以及来源Agent的RAW/PRIVATE；训练标签与规范真值均未输入行动特征。",
"",
"第一处修复保证状态理解正确，第二处修复允许相同状态下不同动作受到不同风险反馈。原MLP骨干和训练协议不变；新模型参数量4471（原4466），显式Horn结构是强归纳偏置，不是模型学会开放域自然语言逻辑。",
"",
"## 2. 2×2因果消融：3 seeds，相同150次总参数更新",
"",
"| Horn传播 | 可训练风险×动作交互 | 测试8世界的合规完成：三个种子 |",
"|---|---|---|",
"| 原错误soft | 无 | 0/8，0/8，0/8 |",
"| 正确零保持 | 无 | 0/8，0/8，0/8 |",
"| 原错误soft | 有 | 4/8，4/8，4/8 |",
"| **正确零保持** | **有** | **8/8，8/8，8/8** |",
"",
"该结果确认两处问题的组合是复现实验中失败的关键因素；不能把表示层修复获得的收益归因给CCE算法。",
"",
"## 3. 修复后五方法真实端到端训练",
"",
"所有方法仍从同一随机种子模型初始化出发，30次共同Action SFT预热，加120次不同目标的更新；训练16世界，开发4世界，探索性测试8世界，知识规则家族完全分开。固定工具读取步骤对所有方法一致。真实R收件历史和业务状态由原生工具逐步更新。",
"",
"| 方法 | 修复前：安全完成，3 seeds | 修复后：安全完成，3 seeds | 修复后：CI违规，3 seeds |",
"|---|---|---|---|",
]
def fmt(vals):
    return "，".join(str(v)+"/8" for v in vals)
for name,label in method_names:
    lines.append("| "+label+" | "+fmt(arr(old,name,"safe_success"))+
                 " | "+fmt(arr(fix,name,"safe_success"))+
                 " | "+fmt(arr(fix,name,"ci_violations"))+" |")
lines+= [
"",
"CCE+DPR的来源纠正版本产生0条可用训练修正对；额外实施的沿真实轨迹向下游B回溯的DPR版本，在三个种子分别找到3、3、3条经原生后续证明的合法同状态纠正对，仍均为8/8合规完成，没有超越普通SFT。额外失败轨迹采样和native参考搜索构成DPR成本。",
"",
f"精确原生规划器在原8个探索测试世界的平均最优回报为{q_mean:.4f}，修复后SFT实际终局回报均值为{sft_return:.4f}，差值为{q_mean-sft_return:.4f}。上游A证据包仅4/8选中长期最优，而下游B在实际历史下的准备动作8/8正确。所以修复解决了合规完成，但没有解决上游多条合法路径的成本最优性。",
"",
"## 4. 追加32个原生世界的post-hoc压力测试",
"",
"仅扩展两个已有测试知识图中的事实置换及业务所需证据组合（V3.1机制变体编号3到6），匹配/额外成本、包名交换后共32新世界，不加入训练。重新训练原训练知识图，冻结后实际执行模型动作：",
"",
"| 方法 | seed7 | seed1234 | seed2025 | 每世界平均原生Regret：三个种子 |",
"|---|---|---|---|---|",
]
for method,label in (("action_sft","Action SFT"),
                     ("cce_reference_dpo_proxy","CCE DPO代理")):
    group=[x for x in stress["model_results"] if x["method"]==method]
    assert len(group)==3
    lines.append("| "+label+" | "+" | ".join(
        str(x["safe_success"])+"/32" for x in group)+" | "+
        "，".join(f"{x['mean_native_regret']:.4f}" for x in group)+" |")
lines+= [
"",
"所有32个native最优参考值被独立重算。六个已保存的小型PyTorch模型权重被校验并重新执行192条完整原生轨迹（2114个真实工具行动转移）；没有任何模型通过读取隐藏测试标签得到答案。测试知识图在修复过程中已被查看，所以这批变体只是事后鲁棒性回归，**不是独立blind benchmark**。",
"",
"## 5. 额外上游公开资源合同比对",
"",
"原始RecordsAgent在选择两种合法证据包时看不到以后可能需要开通保护通道的公开建立费用和单位步骤成本。可单独暴露B随后也能见到的privacy_resource_contract，不传递Qstar、标签或接收者隐藏知识。这个可见信息修补已通过状态转移不变测试；同协议seed7的SFT/CCE虽仍8/8安全完成，A最优选择仍4/8，不能将其称为已解决的长期规划能力。",
"",
"## 6. 科研判断",
"",
"**可以确认：** 旧版方法比较受到表示层缺陷干扰，原0/8并不足以否定SFT或CCE；正确编码与候选交互修复以后，普通SFT已经可以实现多个Agent的安全协作路径。",
"",
"**不能确认：** CCE或DPR相对于普通SFT有独立学习收益。CCE还需精确reference搜索支持，且在压力测试有一个种子28/32不及SFT；当前优点不属于算法。完整共享LLM、自然语言语义推断、Qpi、正式DPO/PPO/GRPO和真正盲测都未完成。",
"",
"**仍待解决：** 可信上游成本优化、多个下游错误间信用分配与有效DPR增量、和更复杂的真实任务。下一轮不能仅增加规则图或训练轮数，而应冻结新的业务机制盲测并以Action SFT与精确规划器作为强基线。",
"",
"## 7. 复现位置",
"",
"- 旧实验和全部原生轨迹仍保留在CHAIN_SHARED_POLICY_METHOD_COMPARISON_01目录。",
"- 修复2×2、源数据和独立回放均在20261120_v31的新目录中；修复仅更改模型编码和特征，不更改环境、任务、已送达历史及Oracle评判。",
"- 主要修复模块：chain_shared_actor_cpu_fixed.py、chain_shared_actor_cpu_action_interaction.py；训练器：compare_kci_chain_methods_cpu_action_interaction.py；新增测试：test_chain_zero_preserving_horn_repair.py、test_chain_action_interaction_repair.py。",
]
DOC.write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n")
paths=[
 B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_01"/"RESULTS.json",
 B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01"/"RESULTS.json",
 B/"CHAIN_SHARED_POLICY_METHOD_COMPARISON_ACTION_INTERACTION_01"/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json",
 B/"CHAIN_DPR_BACKTRACKED_ACTION_INTERACTION_REPAIRED_01"/"RESULTS.json",
 B/"CHAIN_DPR_BACKTRACKED_ACTION_INTERACTION_REPAIRED_01"/"INDEPENDENT_NATIVE_REPLAY_AUDIT.json",
 B/"CHAIN_CPU_HORN_RULE_BY_ACTION_INTERACTION_2X2_01"/"RESULTS.json",
 B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"STRESS.json",
 B/"CHAIN_REPAIRED_POSTHOC_NEW_VARIANTS_01"/"INDEPENDENT_REPLAY_AUDIT.json",
 B/"CHAIN_CPU_HELDOUT_FAILURE_DIAG_01"/"DIAGNOSIS.json",
]
receipt={
 "status":"SOURCE_VERIFIED_CHAIN_ACTOR_TWO_BUG_FIX",
 "old_safe_B_false_risk_rate":[false_alarms,len(safes)],
 "two_by_two_sft_safe_successes":expect,
 "repaired_sft_success_per_seed":arr(fix,"action_sft","safe_success"),
 "repaired_cce_success_per_seed":arr(fix,"cce_reference_dpo_proxy","safe_success"),
 "repaired_terminal_success_per_seed":arr(fix,"terminal_reinforce","safe_success"),
 "repaired_downstream_B_visited_action_per_seed":arr(fix,"action_sft","B_correct_on_visited_history"),
 "reference_native_mean_Qstar":q_mean,
 "repaired_sft_mean_native_return":sft_return,
 "repair_success_not_algorithm_contribution":True,
 "original_data_changed":False,
 "not_fresh_blind_test":True,
 "not_trained_LLM":True,
 "evidence_SHA256":{str(x.relative_to(R)):sha256(x.read_bytes()).hexdigest() for x in paths},
}
META.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="evidence_SHA256"},ensure_ascii=False,indent=2),flush=True)
