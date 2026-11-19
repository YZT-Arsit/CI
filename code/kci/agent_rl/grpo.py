"""Bounded on-policy group-relative policy-gradient trainer for KCI actions.

This implements a clipped GRPO-style *categorical constrained-action* baseline,
not stock TRL GRPO for unconstrained text. Each selected letter is one LM token.
Groups are collected from the same world prefix under ONE frozen-at-collection
policy; zero-variance groups are logged but not used for updates.

No outcome oracle may enter the model prompt. The trajectory terminal result
only supplies a scalar advantage to actual selected action tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

from .core import Episode, group_advantages, group_signal


@dataclass(frozen=True)
class GRPOConfig:
    lr: float = 1e-5
    clip_eps: float = 0.2
    kl_coef: float = 0.02
    entropy_coef: float = 0.0
    grad_clip: float = 1.0
    train_cost_only: bool = False

    def __post_init__(self):
        if not 0 < self.lr < 1e-2:
            raise ValueError("Invalid learning rate")
        if not 0 < self.clip_eps < 1:
            raise ValueError("Invalid policy-ratio clipping range")
        if not 0 <= self.kl_coef <= 1:
            raise ValueError("Invalid KL coefficient")
        if not 0 <= self.entropy_coef <= 1:
            raise ValueError("Invalid entropy coefficient")
        if not 0 < self.grad_clip <= 100:
            raise ValueError("Invalid gradient norm limit")


class GRPOUpdater:
    def __init__(self, policy, cfg: GRPOConfig):
        if not policy.trainable:
            raise ValueError("GRPO requires a trainable LoRA policy")
        self.policy = policy
        self.cfg = cfg
        self.torch = policy.torch
        self.params = policy.trainable_parameters()
        if not self.params:
            raise RuntimeError("LoRA policy has no trainable parameters")
        self.optimizer = self.torch.optim.AdamW(
            self.params, lr=cfg.lr, weight_decay=0.0)
        self.optimizer_steps = 0

    def update(self, group: list[Episode]) -> dict:
        advantages = group_advantages(group)
        rewards = [e.reward for e in group]
        base = {
            "group_id": group[0].scenario_key,
            "sample_size": len(group),
            "rewards": rewards,
            "reward_mean": sum(rewards) / len(group),
            "safe_success": sum(e.success and not e.violation for e in group),
            "violations": sum(e.violation for e in group),
            "optimizer_step": self.optimizer_steps,
        }
        if any(e.policy_id != self.policy.policy_id for e in group):
            raise RuntimeError("Mixed old-policy versions in a GRPO group")
        base["signal_kind"] = group_signal(group)
        if advantages is None:
            return {**base, "updated": False, "skip_reason": "zero_reward_variance"}
        if base["signal_kind"] == "cost_only" and not self.cfg.train_cost_only:
            return {**base, "updated": False,
                    "skip_reason": "cost_only_signal_not_enabled"}
        branch_counts = [
            sum(len(d.actions) > 1 for d in e.decisions) for e in group
        ]
        if sum(branch_counts) == 0:
            return {**base, "updated": False,
                    "skip_reason": "no_trainable_branch_decisions"}

        self.optimizer.zero_grad(set_to_none=True)
        # Enable activation checkpointing while training large QLoRA models.
        # LoRA dropout is fixed at zero; model configuration must also have
        # zero base dropout to preserve rollout/optimization policy identity.
        self.policy.model.train()
        ratios = []
        divergences = []
        entropies = []
        pg_losses = []
        used = 0
        n = len(group)
        for episode, adv, branch in zip(group, advantages, branch_counts):
            if branch == 0:
                continue
            for decision in episode.decisions:
                if len(decision.actions) <= 1:
                    continue
                lp = self.policy.log_distribution(
                    decision.prompt, tuple(decision.actions))
                if not 0 <= decision.chosen_index < len(decision.actions):
                    raise RuntimeError("Corrupt recorded choice index")
                old = self.torch.tensor(
                    decision.old_logprob, device=lp.device, dtype=lp.dtype)
                delta = lp[decision.chosen_index] - old
                ratio = delta.exp()
                if not self.torch.isfinite(ratio):
                    raise RuntimeError("Non-finite importance ratio")
                a = self.torch.tensor(
                    adv, device=lp.device, dtype=lp.dtype)
                pg = -self.torch.minimum(
                    ratio * a,
                    ratio.clamp(1 - self.cfg.clip_eps,
                                1 + self.cfg.clip_eps) * a)
                probabilities = lp.exp()
                entropy = -(probabilities * lp).sum()
                if self.cfg.kl_coef:
                    # Adapter-free reference is a frozen base LM. It does not
                    # share or copy any updating LoRA weights.
                    ref = self.policy.reference_log_distribution(
                        decision.prompt, tuple(decision.actions))
                    kl = (probabilities * (lp - ref)).sum()
                else:
                    kl = lp.sum() * 0.
                weight = 1.0 / (n * branch)
                loss = (
                    pg + self.cfg.kl_coef * kl
                    - self.cfg.entropy_coef * entropy
                ) * weight
                if not self.torch.isfinite(loss):
                    raise RuntimeError("Non-finite GRPO action loss")
                # Each action gets the group terminal advantage; normalize
                # within the episode to avoid long-trajectory dominance.
                loss.backward()
                ratios.append(float(ratio.detach()))
                divergences.append(float(kl.detach()))
                entropies.append(float(entropy.detach()))
                pg_losses.append(float(pg.detach()))
                used += 1

        if used == 0:
            raise RuntimeError("Expected nonzero trainable branch decisions")
        grad_norm = float(
            self.torch.nn.utils.clip_grad_norm_(
                self.params, self.cfg.grad_clip, error_if_nonfinite=True))
        if not math.isfinite(grad_norm):
            raise RuntimeError("Non-finite policy gradient norm")
        self.optimizer.step()
        self.optimizer_steps += 1
        return {
            **base, "updated": True, "optimizer_step": self.optimizer_steps,
            "advantage_mean": sum(advantages) / len(advantages),
            "trainable_action_tokens": used,
            "mean_ratio": sum(ratios) / used,
            "mean_reference_kl": sum(divergences) / used,
            "mean_action_entropy": sum(entropies) / used,
            "mean_surrogate_loss": sum(pg_losses) / used,
            "clipped_ratio_share": sum(
                x < 1 - self.cfg.clip_eps or x > 1 + self.cfg.clip_eps
                for x in ratios) / used,
            "pre_clip_gradient_norm": grad_norm,
        }
