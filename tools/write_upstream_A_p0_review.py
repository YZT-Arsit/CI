"""Generate a source-backed Chinese P0 research decision memo from frozen evidence."""
from pathlib import Path
from collections import defaultdict,Counter
from hashlib import sha256
import json,statistics
R=Path(__file__).resolve().parents[1]
B=R/"artifacts"/"20261120_v31"
A=B/"A_UPSTREAM_DECISION_LABEL_AND_OBSERVABILITY_AUDIT_01"
P=B/"A_PUBLIC_ONLY_COST_PLANNER_P0_01"
S=B/"A_PUBLIC_COST_SFT_CCE_PAIRED_P0_01"
V=B/"A_UPSTREAM_P0_INDEPENDENT_VALIDATION_01"
DOC=R/"docs"/"KCI_UPSTREAM_A_QSTAR_ACTION_BINDING_P0_REVIEW_CN.md"
AUDIT=R/"docs"/"KCI_UPSTREAM_A_QSTAR_ACTION_BINDING_P0_REVIEW_AUDIT.json"
if DOC.exists() or AUDIT.exists():raise FileExistsError("Will not overwrite previous report")
def j(p):return json.loads(p.read_text(encoding="utf-8"))
a=j(A/"RAW_AUDIT.json")
pp=j(A/"PROBABILITIES_TRAINED_SHARED_ACTOR.json")
p=j(P/"RESULT.json")
s=j(S/"P0_A_SFT_VS_QSTAR.json")
v=j(V/"VALIDATION.json")
assert v["status"]=="INDEPENDENT_UPSTREAM_A_P0_SOURCE_AND_RESULT_AUDIT_PASS"
assert len(p["predictions"])==28
assert len(s["all_seed_checkpoints"])==40
raw_test=[x for x in a["rows"] if x["split"]=="exploratory_test"]
assert len(raw_test)==8
assert all(set(x["policy_choices"].values())=={x["risky_package"]} for x in raw_test)
assert sum(x["abs_gap"]==0 for x in raw_test)==2
assert all(x["exact_Qgap"] and x["correct"] for x in p["predictions"])
for name,correct in [
    ("True__optimal_action_SFT",6),
    ("True__Qgap_weighted_SFT",4),
    ("True__Qgap_shuffled_SFT",6),
    ("True__Qgap_reference_pair_DPO_proxy",6),
    ("False__optimal_action_SFT",6),
]:
    assert s["aggregate"][name]["test_strict_correct_per_seed"]==[correct]*5
prob=pp["summary"]
assert prob["('action_sft', 'exploratory_test', 'matched')"]["mean_pred_risky_probability"]>.99
assert prob["('action_sft', 'dev', 'matched')"]["mean_pred_risky_probability"]<.01
sections=[
"# KCI-Align 上游 A 的长期价值识别与行动绑定：P0 决定性审计",
"",
"**研究状态：冻结动态A→R→B→C环境上的CPU P0实验。已复现失败原因、验证A的可观测价值、独立执行公开规则/成本规划器，并进行了同训练状态、五种子动作SFT vs Qstar监督对照。不是4B模型实验；已查看探索测试图，因此不是blind test。**",
"",
"## 1. 上游4/8不是随机无法学习：模型对同一类别的高风险证据过度自信",
"",
"按真实上游源记录（非隐藏预填奖励），当前RecordsAgent读取两个当前均合法的源包；两者会让下游协调Agent后续付出不同隐私费用。审计加载了原修复后的CPU模型小权重，不重复训练，直接获取每次动作概率。",
"",
"在8个探索性测试世界中，三个随机种子一致选择‘将来会要求下游隐私处理’的风险包：matched同成本4个世界全错，tradeoff高安全包成本4个世界里2个严格正确、2个价值打平。故4/8统计掺入2次平局，真正有严格优劣的6个世界仅2/6最优。",
"",
"| 条件 | 世界数 | 真正较优的上游源包 | 原共享策略 |",
"|---|---:|---|---|",
"| 两包当前成本相同 matched | 4 | 安全证据包 | 全部选高风险包，0/4 |",
"| 安全包额外成本4步 tradeoff | 2 | 快速高风险包，随后合法保护 | 全部选高风险包，2/2 |",
"| 价值严格相等（tradeoff） | 2 | 两者都合格 | 选高风险包，两者无regret |",
"",
"在这些测试世界中，非零Qstar行动差由环境实际后续行动得到，其绝对值分别是0.028、0.084和0.092，明显大于数值舍入尺度；不属于接近零的标签噪声。",
"",
"从原共享模型的冻结概率进一步发现：SFT在测试matched中平均给高风险包0.99997的概率，在测试tradeoff中近乎1；但在开发matched中概率近0，在开发tradeoff中也近0。后者在该开发组的tradeoff方向又是错误的。**它在不同已见知识图间形成过度自信、不能跨规则家族迁移的动作价值估计，而非样本本身不可判别。**",
"",
"## 2. A公开观测信息：合法补齐成本合同，但不是问题全部解释",
"",
"原A观察包含每个源包的可用事实及当前处理步数、未来必要的S01-S06检查和公开Horn推理规则，但没有后续隐私通道建立费用与每步成本。独立的source-only观察子类将B后来也可见的资源合同加入A，不新增隐藏环境字段或前向Qstar标签。原本的原生业务状态、真实发送与收件人账本保持完全一致。",
"",
"同一冻结模型在推理时仅补公开合同，引起的A候选概率均值变化接近0；先前在seed7用补合同重新训练也仍只有4/8上游最优。**缺少公开费用是观测合同缺陷，但并非足以解释4/8的完整原因；表示结构、少数A监督样本和规则家族泛化同时影响。**",
"",
"## 3. 不依赖隐藏Qstar和未来原生搜索的强基线已经精确解出当前任务",
"",
"构造独立、完全可解释的A角色公开信息规划器：只读真实源包、下游必要检查、公开规则、资源合同；逐步判断R是否会因每一次必需证据的直接披露构成受保护关联，计算必须使用多少次private attestation和preflight setup，最后比较两合法证据包的变量成本。",
"",
"该规划器**不访问eval-only Qstar、隐藏知识真值和任何未来native穷举**，仅以原生Qstar作为实验结束后的验证。训练16/16、开发4/4、探索测试8/8最优；并且**28/28的两个候选动作Qstar差值数值也完全一致**。",
"",
"这是一条强而且对方法主张不利的结果：当前合成任务的长期A排序可以由公开规则和确定性业务成本解析求出。精确CCE native未来穷举并不是取得这组偏好所必需的唯一途径。",
"",
"## 4. 成对可训练CPU小策略：普通最优行动监督已经够用",
"",
"从同一批初始A源观察出发，训练只使用四类训练图中的14个严格不同价值的状态（原16例中2条平局不强行指定单一正确动作），开发4严格状态，测试6严格状态加2平局。",
"",
"为避免候选包名称的词面捷径，使用对PACKAGE_A和PACKAGE_B共享权重的线性动作打分器；行动特征来自合法公开规则、源包成本、可见下游setup费用。‘full’版本额外含按公开规则推算需要私下认证的次数，这是非常强的手写规划归纳偏置，**不等同LLM自然语言自主规划**。",
"",
"相同5个配对seed、同一个初始化、相同14个训练状态、Adam及80次更新，比较最优动作标签CE、真实Qstar差值加权CE、随机打乱权重CE和相对初始参考的Qstar成对偏好DPO代理。",
"",
"| 轻量A动作学习（公开成本特征） | 严格6条未见图动作：5 seeds | 全8条均值regret |",
"|---|---:|---:|",
]
for m,label in [
    ("True__optimal_action_SFT","普通动作SFT标签"),
    ("True__Qgap_weighted_SFT","真实Qstar-gap加权SFT"),
    ("True__Qgap_shuffled_SFT","打乱Qstar-gap权重的SFT"),
    ("True__Qgap_reference_pair_DPO_proxy","Qstar参考相对DPO代理"),
]:
    x=s["aggregate"][m]
    sections.append("| "+label+" | "+",".join(str(y)+"/6" for y in x["test_strict_correct_per_seed"])
        +" | "+",".join(str(y) for y in x["test_mean_regret_per_seed"])+" |")
sections+=[
"",
"**普通SFT与打乱价值差权重的SFT五种子均6/6正确；真实Qstar-gap加权CE五种子均4/6，平均regret=0.007；成对DPO代理6/6但没有超过普通SFT。**",
"",
"另一组消融移除私下验证次数的完整可见规划特征，只保留源处理成本、是否未来形成受限关系、公开保护通道建立费和必需检查数量，普通SFT同样在5种子6/6。这说明至少本数据的A选择并不需要额外精确Qstar幅度才能训练出正确行动。",
"",
"注意：这不是完整多Agent共享模型的SFT胜出论证；这里训练的是仅选择A包的一个小线性打分器，并借助公开规则计算特征。它只是一个判断是否值得继续宣称CCE独立必要性的**关键负对照**。普通最优动作标签的获取仍需相应参考或人工监督，不能简单说其总监督成本比CCE免费。",
"",
"## 5. 数据规模与统计意义",
"",
"28个源世界来自7个独立知识图/任务根（4 train、1 dev、2 exploratory test），每根仅做matched/tradeoff及交换PACKAGE_A/B名称，共四个受控派生状态；14条train严格状态并非14个独立业务机制。2个探索图家族早已参与开发/故障定位，因此5个优化随机种子不等于新的5个独立测试世界。",
"",
"目前的正反反馈都是真实native轨迹的Qstar参考值，不是实际Qwen策略下的Qpi。读公开规则推算未来认证次数属于强手工先验。当前Task Success在下游能够8/8之后已饱和，唯一剩余成本差别对模型训练方法的辨识力很弱。",
"",
"## 6. 对研究路线的直接审批",
"",
"**P0（已完成）：** 对当前冻结A任务，现有两个候选动作的价值差可靠，可观察；模型未正确学习A是成本与行动绑定/跨图泛化的问题。采用合规可见的规则/成本特征后普通SFT解决了所有严格测试项。当前未见CCE价值加权独立收益。",
"",
"**P1（建议降级）：** DPR只在真失败轨迹的下游违规状态提供局部有效纠正。此前修复后的合成共享CPU策略已饱和，DPR没有再提升SafeTaskSuccess或Regret，暂时视为有用的数据构造/定位工具，不强行绑定CCE核心方法。",
"",
"**P2（必须新机制+封存）：** 本测试只证明有限语法+已知公开规则+确定性成本的A行动可以通过可见规划和SFT学习。下一阶段预先冻结新的业务/角色依赖、真实多阶段可变任务路径和不由同一规则表直接求解的未见情形，避免换事实编号充当独立泛化。预算匹配、Action SFT、显式规划器和真实Terminal RL应是预注册基线。",
"",
"**停止条件：** 如果在充分可见信息、同模型/监督预算以及真正独立的未见任务上，普通Action SFT仍能在Regret和样本效率上追平CCE，就不能以当前‘精确长期Qstar差值’作为方法创新，应考虑转向方法更明确的小问题或建立更强的动态CI评测贡献。",
"",
"**不建议目前租用4B GPU**；应先决定当前方法是否具有独立机制价值，而不是继续训练一个更大的规则分类器。",
"",
"## 实验溯源",
"",
"- A_UPSTREAM_DECISION_LABEL_AND_OBSERVABILITY_AUDIT_01/RAW_AUDIT.json 与 PROBABILITIES_TRAINED_SHARED_ACTOR.json：读取真实冻结模型动作、概率及合法观测。",
"- A_PUBLIC_ONLY_COST_PLANNER_P0_01/RESULT.json：28个同源世界逐项的公开知识规则推导和Qstar数值吻合。",
"- A_PUBLIC_COST_SFT_CCE_PAIRED_P0_01/P0_A_SFT_VS_QSTAR.json：40个种子×监督/特征组合的全部行动概率与训练曲线。",
"- A_UPSTREAM_P0_INDEPENDENT_VALIDATION_01/VALIDATION.json：按原生来源重建，独立交叉验证标签、平局与训练集隔离。",
]
DOC.write_text("\n".join(sections)+"\n",encoding="utf-8",newline="\n")
receipt={
 "status":"KCI_A_UPSTREAM_P0_RESEARCH_DECISION_SUPPORTED_BY_REPLAY",
 "original_dynamic_dataset_modified":False,
 "graph_family_root_counts":{"train":4,"dev":1,"exploratory_test":2},
 "test_strict_comparisons":6,"test_ties":2,
 "raw_shared_policy_strict_success":2,
 "public_visible_planner_Q_gap_exact_instances":28,
 "A_only_SFT_strict_success_5_seed":[6]*5,
 "A_only_Qstar_weighted_strict_success_5_seed":[4]*5,
 "A_only_Qstar_DPO_proxy_strict_success_5_seed":[6]*5,
 "main_CCE_independent_gain_demonstrated":False,
 "new_LLM_training":False,
 "blind_test":False,
 "supporting_sha256":{
    str(x.relative_to(R)):sha256(x.read_bytes()).hexdigest()
    for x in (
       A/"RAW_AUDIT.json",A/"PROBABILITIES_TRAINED_SHARED_ACTOR.json",
       P/"RESULT.json",S/"P0_A_SFT_VS_QSTAR.json",V/"VALIDATION.json")}
}
AUDIT.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n",encoding="utf-8",newline="\n")
print(json.dumps({k:v for k,v in receipt.items() if k!="supporting_sha256"},ensure_ascii=False,indent=2),flush=True)
