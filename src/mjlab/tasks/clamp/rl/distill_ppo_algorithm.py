from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO


class DistillPPO(PPO):
  """PPO with an additional teacher imitation loss on student mean actions."""

  def __init__(
    self,
    policy,
    *args,
    bc_coef_start: float = 1.0,
    bc_coef_end: float = 0.0,
    bc_anneal_iters: int = 10_000,
    bc_loss_type: str = "mse",
    **kwargs,
  ):
    super().__init__(policy, *args, **kwargs)
    self.bc_coef_start = float(bc_coef_start)
    self.bc_coef_end = float(bc_coef_end)
    self.bc_anneal_iters = int(bc_anneal_iters)
    self.num_bc_updates = 0

    if bc_loss_type == "mse":
      self._bc_loss_fn = nn.functional.mse_loss
    elif bc_loss_type == "huber":
      self._bc_loss_fn = nn.functional.huber_loss
    else:
      raise ValueError(
        f"Unknown bc_loss_type: {bc_loss_type}. Supported: ('mse', 'huber')."
      )

  def _current_bc_coef(self) -> float:
    if self.bc_anneal_iters <= 0:
      return self.bc_coef_end
    progress = min(max(self.num_bc_updates / float(self.bc_anneal_iters), 0.0), 1.0)
    return self.bc_coef_start + (self.bc_coef_end - self.bc_coef_start) * progress

  def update(self):  # noqa: C901
    if self.rnd is not None or self.symmetry is not None:
      raise NotImplementedError(
        "DistillPPO currently supports CLAMP PPO setup without RND/symmetry."
      )

    if not hasattr(self.policy, "teacher_act_inference"):
      raise ValueError(
        "DistillPPO requires policy.teacher_act_inference(obs) to compute BC targets."
      )

    bc_coef = self._current_bc_coef()
    if bc_coef > 0.0 and not getattr(self.policy, "loaded_teacher", False):
      raise ValueError(
        "Teacher weights are not loaded. "
        "Resume from a CLAMP distillation checkpoint (or a distill-RL checkpoint) "
        "before running DistillPPO."
      )

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_bc_loss = 0.0

    if self.policy.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for (
      obs_batch,
      actions_batch,
      target_values_batch,
      advantages_batch,
      returns_batch,
      old_actions_log_prob_batch,
      old_mu_batch,
      old_sigma_batch,
      hid_states_batch,
      masks_batch,
    ) in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          advantages_batch = (advantages_batch - advantages_batch.mean()) / (
            advantages_batch.std() + 1e-8
          )

      self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
      actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
      value_batch = self.policy.evaluate(
        obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
      )
      entropy_batch = self.policy.entropy
      mu_batch = self.policy.action_mean
      sigma_batch = self.policy.action_std

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = torch.sum(
            torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
            + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
            / (2.0 * torch.square(sigma_batch))
            - 0.5,
            axis=-1,
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(
        actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
      )
      surrogate = -torch.squeeze(advantages_batch) * ratio
      surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (returns_batch - value_batch).pow(2).mean()

      if bc_coef > 0.0:
        with torch.no_grad():
          teacher_actions = self.policy.teacher_act_inference(obs_batch)
        student_actions = self.policy.act_inference(obs_batch)
        bc_loss = self._bc_loss_fn(student_actions, teacher_actions)
      else:
        bc_loss = torch.zeros((), device=self.device)

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy_batch.mean()
        + bc_coef * bc_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy_batch.mean().item()
      mean_bc_loss += bc_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_bc_loss /= num_updates

    self.storage.clear()
    self.num_bc_updates += 1

    return {
      "value_function": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "bc": mean_bc_loss,
      "bc_coef": bc_coef,
    }
