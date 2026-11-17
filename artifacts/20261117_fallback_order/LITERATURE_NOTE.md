# Source qualification

The original CLEAR paper studies stability/plasticity using experience replay in multi-task reinforcement learning: [Rolnick et al., Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682). It motivates asking whether learned behavior survives later updates.

This local experiment is a supervised LoRA schedule intervention. It does not reproduce CLEAR, does not add its behavioral-cloning or off-policy losses, and cannot inherit its empirical claims. Changes in the present matched schedules are evidence about these model/data/optimizer conditions only. Adam state changes with order; parameter interference is not separately identified. A failing final checkpoint alone does not establish forgetting or insufficient model capacity.
