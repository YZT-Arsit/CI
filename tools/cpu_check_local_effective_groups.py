from diagnose_state_matched_cpu import OUT
import json
d=json.loads((OUT/"CPU_CREDIT_DIAGNOSTIC.json").read_text(encoding="utf-8"))
g=d["credit_mechanism"]["groups"]
for x in g:
    if x["pCCE_supported"]:
        print(json.dumps({
            "group":x["group"],"role":x["role"],"pair_id":x["pair_id"],"related":x["related"],
            "A_grad":x["A_grad_norm"],"B_grad":x["B_grad_norm"],
            "A_tokens":x["A_action_tokens"],"B_tokens":x["B_action_tokens"],
            "same_train_first_action_sample_count":8-x["different_first_action"],
            "same_train_full_trajectory_count":8-x["different_path"],
            "different_train_outcomes":x["different_terminal_outcome"],
            "A_train_violations":x["A_train_violations"],"B_train_violations":x["B_train_violations"],
        },ensure_ascii=False))
