r"""CPU analytic sensitivity on actual four 4B Qstar-flip model-misranked states.
This is a first-order gradient coefficient w.r.t action logit margin only,
NOT a LoRA parameter update or proof that DPO labels are valid.
"""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"artifacts"/"20261119_composition_signals"/"agent_rl_v1"/"policy_qpi_01"
OUT=BASE/"cpu_rl_family_pilot_20261119"
a=json.loads((BASE/"STATE_HISTORY_VISIBLE_ACTION_FLIP_AUDIT.json").read_text(encoding="utf-8"))
assert a["summary"]["oracle_best_flipping_pairs"]==4
eps=.35
def sigmoid(x):
    return 1/(1+math.exp(-x))
records=[]
for x in a["pairs"]:
    if not x["oracle_best_flip"]:continue
    assert x["model_pref_related"]!=x["oracle_best_related"]
    # For the observed related history state, model prefers action A, with
    # positive raw A-vs-B margin, while true Qstar ideal path favors other.
    m=x["model_raw_margin_A_related"]
    assert m>0 and x["model_pref_related"]==x["model_pref_unrelated"]
    deriv={}
    for T in (1,2,4,8):
        s=sigmoid(m/T)
        p_good=eps/2+(1-eps)*(1-s)
        grad=(1-eps)*s*(1-s)/(T*p_good)
        deriv[str(T)]={"prob_Qstar_preferred":p_good,
                        "d_logpi_good_d_correct_logit_margin":grad}
    # Paired DPO with reference snapshot identical to current policy,
    # beta=.1, raw *unmixed* policy log-probs (not eps-mixture sampling).
    # At theta = ref the pairwise logit-margin loss derivative is -beta/2.
    records.append({"pair_id":x["pair_id"],"actor":x["role"],
                    "wrong_action_margin":m,"by_temperature":deriv,
                    "raw_pairwise_logit_DPO_initial_gradient_magnitude_beta_0_1":.05})
stat={
    "four_wrong_policy_argmax_flip_pairs":len(records),
    "epsilon_sampling":eps,
    "finite_beta_0_1_raw_logit_DPO_initial_derivative":.05,
    "average_correct_action_logprob_gradient":{
        str(T):statistics.mean(r["by_temperature"][str(T)]["d_logpi_good_d_correct_logit_margin"]
                               for r in records)
        for T in (1,4)},
    "records":records,
    "warnings":[
        "The correct action here is Qstar-optimal under a perfect continuation, not necessarily the best Qpi action under frozen 4B.",
        "A preference derived from Qpi or native evidence is required before DPO training; the Qstar action should not be assumed a valid deployment target.",
        "DPO gradient 0.05 assumes beta=.1 and UNSMOOTHED raw two-action log-softmax preference ratio, with initial reference=current policy.",
        "If DPO is instead computed from the epsilon-mixture sampling policy, it can suffer the same saturation.",
        "Step-PPO and GiGPO still use sampled action log-prob and can remain saturated at T1; changing optimizer name does not fix this.",
        "Neither scalar derivative is the actual neural LoRA gradient, which depends on the Jacobian of logits."
    ],
}
p=OUT/"GRADIENT_FAMILY_ANALYTIC.json"
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(stat,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print(json.dumps(stat,ensure_ascii=False,indent=2))
