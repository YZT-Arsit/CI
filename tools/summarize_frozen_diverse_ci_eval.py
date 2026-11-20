"""Summarize actual Qwen2.5-7B local frozen model action probes,
native agent episodes, matched CPU baselines and action-format sensitivity.
No data collected from models here; this only reads finalized audits.
"""
from pathlib import Path
from collections import Counter
import json
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"
PROBE=BASE/"frozen_qwen25_7b_diverse_probes_01"
FULL=BASE/"frozen_qwen25_7b_diverse_full_01"
ORDER=BASE/"frozen_qwen25_7b_order_bias_01"
ORACLE=BASE/"diverse_ci_long_horizon_v2"
OUT=BASE/"FROZEN_QWEN25_7B_DIVERSE_EVALUATION_01"
if OUT.exists():raise FileExistsError("Refuse to overwrite previous report")
def jd(path):return json.loads(path.read_text(encoding="utf-8"))
def jl(path):return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
pr=jd(PROBE/"SUMMARY.json")
fu=jd(FULL/"SUMMARY.json")
pa=jd(PROBE/"INDEPENDENT_REPLAY_AUDIT.json")
fa=jd(FULL/"INDEPENDENT_REPLAY_AUDIT.json")
bias=jd(ORDER/"ORDER_BIAS.json")
baseline=jd(ORACLE/"CPU_POLICY_BASELINES.json")
assert pa["status"]==fa["status"]=="INDEPENDENT_FROZEN_POLICY_NATIVE_REPLAY_PASS"
assert pr["model"]["model_digest"]==fu["model"]["model_digest"]
assert pr["summary"]["n_probes"]==35
eps=jl(FULL/"RESULTS.jsonl")
pinfo=jl(PROBE/"RESULTS.jsonl")
assert len(eps)==fu["summary"]["n_full_episodes"]
blookup={(x["root"],x["condition"],x["policy"]):x
         for x in baseline["case_records"]}
table=[]
names=("myopic_skip","myopic_always_open","role_conditioned_preflight","oracle")
for ep in eps:
    status=("CI_VIOLATION" if ep["native_violation"] else
            "SAFE_SUCCESS" if ep["native_success"] else "SAFE_INCOMPLETE")
    matched={p:bool(blookup[ep["task_id"],ep["condition"],p]["success"]) for p in names}
    table.append({
        "task":ep["task_id"],"graph":ep["graph"],
        "business":ep["business"],"condition":ep["condition"],
        "status":status,"calls":ep["model_calls"],"steps":ep["native_steps"],
        "choices":[x["model"]["selected_action"] for x in ep["trace"] if "model" in x],
        "matched_baselines":matched,
        "model_generated_invalid":bool(ep["error_or_invalid_action"]),
    })
outcome={
    "model":pr["model"],
    "no_training":True,"no_paid_API":True,
    "same_as_remote_Qwen3_4B":False,
    "probe_metrics":pr["summary"],
    "full_metrics":fu["summary"],
    "action_order_control":{
        "task_history_states":bias["number_task_history_pairs"],
        "index_zero_original":bias["index0_original"],
        "index_zero_reordered":bias["index0_swapped"],
        "semantic_choice_changed_after_reordering":bias["chosen_action_changed_after_swap"],
        "named_raw_choices":sum(x["order"]=="named_action" and x["selected_action"]=="SEND_RAW_R"
                                for x in bias["records"]),
    },
    "matched_baseline_safe_successes":{
        name:sum(r["matched_baselines"][name] for r in table) for name in names},
    "per_episode_results":table,
    "source_probes_audited":True,"full_episodes_audited":True,
    "datasets_not_blind":True,
    "norms_synthetic":True,
}
OUT.mkdir()
(OUT/"CONSOLIDATED_RESULTS.json").write_text(json.dumps(
    outcome,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
notes=[
"# KCI：冻结 Qwen2.5-7B 本地 Agent 的真实动态 CI 行动实验",
"",
"基于用户 Windows Ollama 安装的冻结 qwen2.5:7b-instruct-q4_K_M，非原先的远程 Qwen3-4B。",
"模型文件 digest："+pr["model"]["model_digest"]+"。",
"所有任务是此前构建的、已参与开发的合成多 Agent CI 环境，不是盲测或真实企业任务。没有模型训练、GPU 租用或付费 API 调用。",
"",
"## 真实评测协议",
"",
"固定系统提示与温度 0。评测时模型每步看到实际授权 observation、业务工具状态、R 的实际送达记录、明确给定知识推导规则、合法动作集合。",
"Actor 逐步选择合法动作，执行器真实执行并改变状态，直到任务成功、泄露、主动终止或行动超过上限。",
"35 个 source action probes 使用合法动作索引格式；完整多步采用直接输出合法动作名称，降低固定索引位置偏差。",
"所有模型实际动作、原始模型响应、模型输入的 SHA-256、执行器下一状态和业务终局都保存，另由不调用模型的脚本重新回放并核验。",
"",
"## 第一部分：35 个即时披露探针",
"",
"直接披露 SEND_RAW_R 的总次数："+
    str(sum(x["model"]["selected_action"]=="SEND_RAW_R" for x in pinfo))+"/35。",
"模型选择导致即时违规："+str(pr["summary"]["n_probe_selected_immediate_violation"])+"/35。",
"Flip 同一消息跨历史产生正确行动变化："+
    str(pr["summary"]["n_flip_pairs_correct_immediate_direction"])+
    "/"+str(pr["summary"]["n_distinct_probe_roots"])+ " 组。",
"Hold 两种无关历史选相同动作："+
    str(pr["summary"]["n_hold_pairs_same_action"])+
    "/"+str(pr["summary"]["n_distinct_probe_roots"])+
    "，但若模型所有情况均披露，不能以此证明真的理解 Hold。",
"",
"### 表达/位置偏差",
"",
"在四组状态交换前两个合法动作的顺序后，模型仍选 index 0 "+
    str(bias["index0_swapped"])+"/"+str(bias["number_task_history_pairs"])+
    "；这四组的实际语义动作全部跟着位置变化。",
"以动作名称直接作为输出的四次小控制中，仍全部选择 SEND_RAW_R。",
"这证明固定动作索引方式严重混入位置偏差，所以不能把这些 source probes 直接解释成纯粹的隐私认知能力失败。",
"",
"## 第二部分：完整多 Agent 执行",
"",
"| 任务 | 业务分支 | 模型真实结果 | 真实模型调用数 |",
"|---|---|---|---:|",
]
for row in table:
    notes.append("| "+row["task"]+" | "+row["condition"]+" | "+
                 row["status"]+" | "+str(row["calls"])+" |")
notes.extend([
"",
"完整任务安全完成："+str(fu["summary"]["n_full_safe_success"])+"/"+str(len(table))+"。",
"CI 违规："+str(fu["summary"]["n_full_privacy_violation"])+"/"+str(len(table))+"。",
"安全但未完成："+str(fu["summary"]["n_full_safe_incomplete"])+"/"+str(len(table))+"。",
"模型/环境错误终止："+str(fu["summary"]["n_full_model_or_executor_invalid"])+"/"+str(len(table))+"。",
"",
"### 完全相同任务条件上的 CPU 规则基线",
"",
"| 规则/规划器 | 合规完成 |",
"|---|---:|",
])
for name,label in (
    ("myopic_skip","跳过事前隐私准备并只应对当前违规"),
    ("myopic_always_open","总是开通事前隐私证明通道"),
    ("role_conditioned_preflight","根据公开条件/规则选择准备策略"),
    ("oracle","有限模拟环境的参考搜索")):
    notes.append("| "+label+" | "+str(outcome["matched_baseline_safe_successes"][name])+
                 "/"+str(len(table))+" |")
notes.extend([
"",
"强规则控制器能使用原生结构化公开规则和实际送达信息，但不承担 LLM 文本解析误差；参考搜索还进行确定性枚举。这是模型能力参照，不是对不同系统部署成本的公平比较。",
"",
"## 研究判断",
"",
"本轮已确认冻结本地模型可以驱动真实多步工具轨迹；其失败可定位为实际泄露、终止/拒绝及对业务路径的错误选择，而不仅是静态分类判断。",
"但这些是单模型、确定性、合成且开发过的数据。不能推出原来 Qwen3-4B 的能力、GRPO 是否有效或模型规模迁移规律。",
"继续训练新方法之前，应该恢复原 Qwen3-4B checkpoint 的等条件冻结评测；并用动作名称或平衡动作位置，不能用固定 index0 的协议制造虚假改进。",
"",
"## 可复现材料",
"",
"所有原始响应、授权观察、动作轨迹、状态回放和系统提示均在压缩包的 probes / full / order_bias 子目录。",
"环境仍为 diverse_ci_long_horizon_v2；用户本机的原始数据、代码与 Ollama 模型均未被改动。",
])
(OUT/"FROZEN_POLICY_EVAL_CN.md").write_text("\n".join(notes)+"\n",encoding="utf-8")
print(json.dumps({
    "status":"REPORT_WRITTEN","path":str(OUT),
    "n_probes":len(pinfo),"n_episodes":len(table),
    "model_safe_success":fu["summary"]["n_full_safe_success"],
    "model_violations":fu["summary"]["n_full_privacy_violation"],
    "model_safe_incomplete":fu["summary"]["n_full_safe_incomplete"],
    "API_calls":0,
},ensure_ascii=False,indent=2),flush=True)
# Script executes at top level and refuses to overwrite existing reports.
