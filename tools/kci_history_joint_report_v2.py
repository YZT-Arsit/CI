"""Write reproducible scientific report from verified V2 experimental artifacts.

NO model inference/training or DEV access until the nine training checkpoints
and native DEV evaluation are frozen. Uses only frozen artifacts, decision
gate and static visible-only diagnostic, never adjusts experiment rules.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"artifacts/20261124_joint_history_supervision_v2"
REPORT=ROOT/"docs/KCI_STATE_CONDITIONED_JOINT_HISTORY_3ARM_20261124_CN.md"
METHODS=("action_ce","same_state_dpo","joint_flip_hold_ce")
VIEWS=("original","entity_alias")
SEEDS=(7,1234,2025)
PAIRS=("Flip","Hold","Authorization","Equivalent")
def j(path):return json.loads(Path(path).read_text(encoding="utf8"))
def jl(path):return [json.loads(x) for x in Path(path).open(encoding="utf8") if x.strip()]
def sha(path):
    h=sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()
def fmt1(x):return "NA" if x is None else f"{x:.2f}"
def main():
    if REPORT.exists():raise FileExistsError(REPORT)
    prereg=j(D/"THREE_ARM_PRE_REGISTERED.json")
    fz=j(D/"SET_VALUED_CORRECTION_AUDIT.json")
    isolated=j(D/"TRAIN_ROLE_VISIBLE_PAIR_ISOLATION_AUDIT.json")
    criterion=j(D/"DEV_ELIMINATION_PRE_REGISTERED.json")
    dev=j(D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json")
    elimination=j(D/"JOINT_CANDIDATE_ELIMINATION.json")
    local=j(D/"VISIBLE_ONLY_STATIC_CLOSURE_DIAGNOSTIC.json")
    traces=jl(D/"REAL4B_DEV_FULL_NATIVE/REAL_NATIVE_MODEL_TRAJECTORIES.jsonl")
    assert len(traces)==270 and dev["n_three_arm_real_dev_native_episodes"]==270
    assert prereg["same_data_sha256"]==sha(D/"TRAIN_FIVE_CONTEXTS.jsonl")
    assert prereg["dev_data_sha256"]==sha(D/"DEV_FIVE_CONTEXTS.jsonl")
    assert elimination["source_DEV_results_sha256"]==sha(D/"REAL4B_DEV_FULL_NATIVE/SUMMARY.json")
    assert prereg["all_arms_same_set_valued_native_witness_preference"]
    assert isolated["all_pairs_isolate_exactly_one_actor_visible_field"]
    assert local["n_dev_true_states"]==15
    for arm in METHODS:
      for seed in SEEDS:
        spec=j(D/f"train_{arm}_seed{seed}/TRAINING.json")
        assert spec["training_data_sha256"]==prereg["same_data_sha256"]
        assert spec["init_shared_full_flow_SFT_model_adapter_SHA256"]==prereg["same_init_adapter_SHA256"]
        assert spec["updates"]==28 and spec["n_true_unique_training_states"]==35
        assert spec["set_valued_native_labels_from_equal_CI_success_cost"]
    lines=[]
    ap=lines.append
    ap("# KCI-Align：状态条件化联合历史监督（真实 Qwen3-4B，三方法对照）")
    ap("")
    ap("**科研结论："+("暂保留（仅探索，仍需新的真正独立任务根）" if elimination["decision"]=="PROVISIONAL_RETAIN_FOR_NEW_INDEPENDENT_TEST" else "依预注册淘汰标准，联合监督候选 C 未获支持")+"。**")
    ap("")
    ap("本研究是既有 `tools/kci_v4_history_contrast_train.py` 的受控扩展；**不是新提出的全新算法，也不是 CCE/DPR 训练**。不因为训练集或 Flip 某一指标单独变好而夸大结果。")
    ap("")
    ap("## 1. 数据、共用模型和实验协议")
    ap("")
    ap(f"- 训练业务根 **7**（P201、P202、P203、P205、P206、P207、P208），每根原生 **5** 个状态，**35** 个唯一真实角色可见观察；开发业务根 **3**（P204、P209、P210），**15** 个原生状态，另有角色别名视图，不作为训练增量样本。最终测试根未读取。")
    ap(f"- TRAIN 四类关系：7 对历史 Flip、7 对无关 Hold、7 对授权变化、7 对等价历史，已核对 **28/28** 配对仅隔离原生接收历史或公开授权字段。`{D/'TRAIN_ROLE_VISIBLE_PAIR_ISOLATION_AUDIT.json'}`。")
    ap("- 三臂：A 完整历史 Action CE；B 同状态接受/拒绝 DPO（冻结**同一初始 SFT adapter** 为参考）；C 在相同 CE 之上加入历史 Flip contrast、授权 contrast、Hold/等价历史分布双向 KL。无跨历史 DPO 伪标签。")
    ap(f"- 三臂共用 **Qwen3-4B、同一初始共享 LoRA r8/alpha16、学习率 2e-5、28 次根级更新、每次同根5个状态**；3个训练采样 seed `7,1234,2025`。所有原生候选动作（包括 ABORT、合法中介）保留。初始 adapter SHA256：`{prereg['same_init_adapter_SHA256']}`。")
    ap("- 旧 V1 错误：P206/P207 危险历史中 private VERIFY 和经授权 broker **都是原生安全、成功且成本相同**，不能强制 VERIFY 唯一正确；经独立原生审计后，在任何 DEV 模型评估前冻结 V2 的可接受动作集合；C/CE/DPO 同样修复，V1 旧模型留档但没有进入正式比较。P210 DEV 的 broker 也归为可接受。")
    ap("- **证据限制**：P201/P203/P205/P208 紧期限风险场景中，即时核验合法，但后续可能超时/不能完成任务，当前正确行动不能与合规成功混淆。所用固定 native suffix 是原生成功见证，不是 Qstar 或当前模型 Qπ；真正团队结局单独由模型在原生环境实际选择后续动作测量。")
    ap("- B 的 DPO 偏好仅对已有原生来源披露候选中的优/劣行动建立 pair（RAW/VERIFY/合法中介），仍允许模型实际选择 ABORT，但没有额外创造 ABORT 的 DPO 负标签；CE 则在全部可执行动作（含 ABORT）的 softmax 上训练。B 与 A/C 的优化目标差异须明确报告。")
    ap("- 三个 seed 只改变同一初始共享 SFT 适配器之后的训练顺序/运行随机性；**不等于三份独立 SFT 预训练模型**，不能夸大为模型初始化多样性。")
    ap("- DPO 在同样可微训练预算以外有初始化冻结参考前向，费用单独列示，不称为完全相同 GPU 成本。")
    ap("")
    ap("## 2. 未见业务根——严格整对结果和真实任务后果")
    ap("")
    ap("每个方法、每个视图共有 **3 个训练种子 × 3 个 DEV 根 × 5 个条件=45 条真实原生团队续行**；每种关系最多9个整对。精确最大分数并列严格标记未决，绝不按字典顺序当正确。评价从原生已验证的 source release 状态开始，然后由同一共享4B选择**所有后续角色**可执行动作；不是从原始 A 开始的完全自主任务，也不是固定脚本续行。")
    ap("")
    ap("| 方法 | 视图 | 状态正确 /45 | Flip /9 | Hold /9 | 授权 /9 | 等价 /9 | 合法 RAW /36 | 危险 RAW /9 | 原生合规成功 /45 | 真实CI违规 /45 | 中止/超时/未知 |")
    ap("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    names={"action_ce":"A: Action CE","same_state_dpo":"B: 同状态 DPO","joint_flip_hold_ce":"C: 联合历史监督"}
    for arm in METHODS:
        for view in VIEWS:
            x=dev["aggregates"][arm][view]
            pairs=x["pairs"]
            ap("| "+names[arm]+" | "+("原始" if view=="original" else "匿名")+
               f" | {x['strict_state_correct']}/45 | "+
               " | ".join(f"{pairs[k]}/9" for k in PAIRS)+
               f" | {x['legal_RAW']}/36 | {x['unsafe_RAW']}/9 | "+
               f"{x['native_safe_task_success']}/45 | {x['native_CI_violations']}/45 | "+
               f"{x['native_aborts']}/{x['native_deadline_miss']}/{x['native_UNKNOWN_or_incomplete']} |")
    ap("")
    ap("### 各业务根与训练种子独立读数")
    ap("")
    ap("| 方法 | seed | 视图 | P204 合规完成/CI违规 | P209 合规完成/CI违规 | P210 合规完成/CI违规 | Flip/Hold/授权/等价整对 | 合法 RAW |")
    ap("|---|---:|---|---|---|---|---|---:|")
    for arm in METHODS:
      for seed in SEEDS:
       for view in VIEWS:
        x=dev["per_seed"][arm][str(seed)][view]
        b=x["by_root"]
        ap("| "+names[arm]+f" | {seed} | "+("原始" if view=="original" else "匿名")+" | "+
           " | ".join(f"{b[root]['safe_task_success']}/{b[root]['CI_violations']}" for root in ("P204","P209","P210"))+
           " | "+"/".join(str(x["pairs_both_correct"][k]) for k in PAIRS)+
           f" | {x['legal_RAW']} |")
    ap("")
    ap("## 3. 候选 C 淘汰规则的独立执行")
    ap("")
    ap(f"- 预注册判决：**{elimination['decision']}**；原子原因：`{', '.join(elimination['reasons'])}`。")
    ap(f"- C 相对 CE 的双视图整对变化：原始 `{elimination['per_view']['original']['C_minus_CE_pair_gain']}`，匿名 `{elimination['per_view']['entity_alias']['C_minus_CE_pair_gain']}`。")
    ap(f"- 3个seed中在两个视图均有正向整对增益的seed数：**{elimination['seeds_positive_both_views']}/3**；三个DEV业务根中跨3seed聚合后在两个视图都有正增益的根数：**{elimination['roots_net_positive_in_both_views_across_3_seeds']}/3**。")
    ap(f"- 准入约束的完整检查：`{elimination['per_view']['original']['C_guardrails']}`（原始），`{elimination['per_view']['entity_alias']['C_guardrails']}`（匿名）。有任意合法RAW、原生任务完成或CI违规恶化，就不得称为正收益。")
    ap("")
    ap("## 4. 静态知识闭包——必要的强反事实对照")
    ap("")
    ap("同一来源消息的即时CI判断可以由合法可见的收件账本、公开蕴含规则、当前签名来源及明确授权确定，不依赖任何未来4B模型或原生Qstar。")
    ap(f"- 静态可见规则在真实训练状态上为 **35/35**；在这批未见DEV业务根上为 **{local['state_action_set_correct']}/{local['n_dev_true_states']}**，仅用允许的 actor observation，不使用隐藏未来。")
    ap("- 如果静态对照在DEV同样较强，即使C提高整对行为，也不能声称需要长期CCE、跨历史强化学习或全新算法才能解决这个局部披露问题。")
    ap("")
    ap("## 5. 训练/计算预算")
    ap("")
    ap("| 方法 | seed | 更新 | 训练样本状态数 | 可微候选前向 | DPO 冻结参考前向 | 训练时间(s) |")
    ap("|---|---:|---:|---:|---:|---:|---:|")
    for arm in METHODS:
        for seed in SEEDS:
            s=j(D/f"train_{arm}_seed{seed}/TRAINING.json")
            ap(f"| {names[arm]} | {seed} | {s['updates']} | {s['n_true_unique_training_states']} | {s['total_differentiable_candidate_forwards']} | {s['reference_initial_candidate_forwards']} | {s['training_wall_seconds']:.2f} |")
    ap("")
    ap("模型评估同样按实际输入 token、候选模型评分前向和角色动作后果计账，详见 `REAL4B_DEV_FULL_NATIVE/SUMMARY.json` 与原始逐行动轨迹。此批实验在 3 个未见业务根上属于探索性根级比较，不构成独立知识函数或论文显著性证明。")
    ap("")
    ap("## 6. 研究裁决")
    ap("")
    if elimination["decision"]=="ELIMINATE_JOINT_HISTORY_CANDIDATE":
        ap("**淘汰候选 C。** 依据预注册的严格标准，它没有在两个文本视图、多个训练种子和至少两个 DEV 业务根上，兼顾真实CI合规与任务成功地稳定优于完整历史CE。停止为维护此方法继续调 margin/LoRA/权重或追加同根模板。旧 history-contrast 只作为负/混合对照留档。")
    else:
        ap("**仅保留作更强独立检验，不接受为方法贡献。** 该实验通过基础守门但根数少、静态基线强，必须转向真正新的业务与知识函数根及语义自然语言变体，再严格比较模型整链任务结果，仍不应直接写论文算法优势。")
    ap("")
    ap("下一轮只有在候选通过当前门槛且相对于静态/公开计划强基线仍有不可替代的改善时，才投入新的对齐训练；否则优先回到有明确长期策略后果、简单静态闭包无法充分排序的真实信号任务。")
    ap("")
    ap("**证据路径：**")
    ap("")
    ap("- `artifacts/20261124_joint_history_supervision_v2/THREE_ARM_PRE_REGISTERED.json`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/DEV_ELIMINATION_PRE_REGISTERED.json`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/SET_VALUED_CORRECTION_AUDIT.json`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/TRAIN_ROLE_VISIBLE_PAIR_ISOLATION_AUDIT.json`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/REAL4B_DEV_FULL_NATIVE/REAL_NATIVE_MODEL_TRAJECTORIES.jsonl`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/REAL4B_DEV_FULL_NATIVE/SUMMARY.json`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/JOINT_CANDIDATE_ELIMINATION.json`")
    ap("- `artifacts/20261124_joint_history_supervision_v2/VISIBLE_ONLY_STATIC_CLOSURE_DIAGNOSTIC.json`")
    REPORT.write_text("\n".join(lines)+"\n",encoding="utf8")
    print("JOINT_SUPERVISION_RESEARCH_REPORT_WRITTEN",REPORT,
          "lines",len(lines),"decision",elimination["decision"],flush=True)
if __name__=="__main__":main()
