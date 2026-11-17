# Secondary comparison against the earlier best shallow checkpoint

Registered while the four-arm worker is still evaluating reused baselines, before its final audit. This is **additional exploratory evaluation**, not an amendment of the frozen four-arm primary experiment.

The primary matched sampler ablation uses the repaired sampler's SFT and full-JS checkpoints. Earlier evidence already showed that sampler-SFT was worse than the older `20261117_binding_factors/repair/two_view_sft` checkpoint on shallow complete episodes. Improving only over that degraded control would be insufficient. Therefore evaluate that fixed earlier checkpoint on the **same new112DEVpresentations**, without training, sampling changes, retries or new worlds.

Checkpoint selection is based on the already published previous results. This secondary arm is not an independently preregistered method comparison; it helps prevent an improvement-over-weak-control claim. Keep its outputs, native audit and counts separate from the four-arm448episode result. No new optimizer updates. Free model-generated estimates and actions only, with the same CPU and generation limits.
