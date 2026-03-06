from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn

from mjlab.tasks.clamp.rl.policy import ClampActorCriticMimic
from mjlab.tasks.clamp.rl.student_policy import ClampStudentActorCritic


class ClampStudentDistillPpoActorCritic(ClampStudentActorCritic):
  """Student actor-critic with a frozen teacher actor for BC-regularized PPO."""

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions: int,
    actor_obs_normalization: bool = True,
    critic_obs_normalization: bool = True,
    actor_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    critic_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    state_dependent_std: bool = False,
    teacher_obs_normalization: bool = True,
    teacher_actor_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    teacher_critic_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    teacher_activation: str = "elu",
    teacher_motion_obs_dim: int = 1240,
    teacher_motion_steps: int = 20,
    teacher_motion_latent_dim: int = 128,
    teacher_motion_proj_channels: int = 60,
    teacher_motion_conv_channels: tuple[int, ...] | list[int] = (40, 20),
    teacher_motion_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    teacher_motion_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    teacher_layer_norm: bool = True,
    teacher_state_dependent_std: bool = False,
    teacher_noise_std_type: str = "scalar",
    teacher_fix_action_std: bool = False,
    teacher_action_std: tuple[float, ...] | list[float] | None = None,
    teacher_share_obs_normalizer: bool | None = None,
    teacher_print_model_structure: bool = False,
    **kwargs,
  ):
    super().__init__(
      obs,
      obs_groups,
      num_actions,
      actor_obs_normalization=actor_obs_normalization,
      critic_obs_normalization=critic_obs_normalization,
      actor_hidden_dims=actor_hidden_dims,
      critic_hidden_dims=critic_hidden_dims,
      activation=activation,
      init_noise_std=init_noise_std,
      noise_std_type=noise_std_type,
      state_dependent_std=state_dependent_std,
      **kwargs,
    )

    if "teacher" not in obs_groups:
      raise ValueError(
        "ClampStudentDistillPpoActorCritic requires obs_groups['teacher']."
      )

    teacher_obs_groups = {
      "policy": tuple(obs_groups["teacher"]),
      "critic": tuple(obs_groups["teacher"]),
    }
    self.teacher = ClampActorCriticMimic(
      obs=obs,
      obs_groups=teacher_obs_groups,
      num_actions=num_actions,
      actor_obs_normalization=teacher_obs_normalization,
      critic_obs_normalization=teacher_obs_normalization,
      actor_hidden_dims=teacher_actor_hidden_dims,
      critic_hidden_dims=teacher_critic_hidden_dims,
      activation=teacher_activation,
      init_noise_std=init_noise_std,
      noise_std_type=teacher_noise_std_type,
      state_dependent_std=teacher_state_dependent_std,
      motion_obs_dim=teacher_motion_obs_dim,
      motion_steps=teacher_motion_steps,
      motion_latent_dim=teacher_motion_latent_dim,
      motion_proj_channels=teacher_motion_proj_channels,
      motion_conv_channels=teacher_motion_conv_channels,
      motion_conv_kernel_sizes=teacher_motion_conv_kernel_sizes,
      motion_conv_strides=teacher_motion_conv_strides,
      layer_norm=teacher_layer_norm,
      fix_action_std=teacher_fix_action_std,
      action_std=teacher_action_std,
      share_obs_normalizer=teacher_share_obs_normalizer,
      print_model_structure=teacher_print_model_structure,
    )
    for param in self.teacher.parameters():
      param.requires_grad = False
    self.teacher.eval()
    self.loaded_teacher = False

  def teacher_act_inference(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return self.teacher.act_inference(obs)

  def train(self, mode=True):
    super().train(mode)
    self.teacher.eval()

  def _load_teacher_actor_from_prefixed_state(self, state_dict) -> None:
    teacher_state_keys = set(self.teacher.state_dict().keys())
    required_actor_keys = {
      key
      for key in teacher_state_keys
      if key.startswith("actor.")
      or key.startswith("actor_motion_encoder.")
      or key.startswith("actor_obs_normalizer.")
      or key in ("std", "log_std")
    }

    actor_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
      if not key.startswith("teacher."):
        continue
      mapped_key = key[len("teacher.") :]
      if (
        mapped_key.startswith("actor.")
        or mapped_key.startswith("actor_motion_encoder.")
        or mapped_key.startswith("actor_obs_normalizer.")
        or mapped_key in ("std", "log_std")
      ):
        actor_state_dict[mapped_key] = value

    if len(actor_state_dict) == 0:
      raise ValueError(
        "No teacher actor-side parameters found in checkpoint (expected `teacher.*`)."
      )

    provided_actor_keys = set(actor_state_dict.keys())
    missing_actor_keys = sorted(required_actor_keys - provided_actor_keys)
    if len(missing_actor_keys) > 0:
      raise ValueError(
        "Teacher actor checkpoint is incomplete. Missing required actor-side keys: "
        f"{missing_actor_keys[:20]}"
      )

    unexpected_actor_keys = sorted(provided_actor_keys - teacher_state_keys)
    if len(unexpected_actor_keys) > 0:
      raise ValueError(
        "Teacher actor checkpoint contains unexpected keys for this architecture: "
        f"{unexpected_actor_keys[:20]}"
      )

    incompatible = self.teacher.load_state_dict(actor_state_dict, strict=False)
    remaining_actor_missing = [
      key for key in incompatible.missing_keys if key in required_actor_keys
    ]
    if len(remaining_actor_missing) > 0:
      raise ValueError(
        "Failed to load all required teacher actor parameters. Missing after load: "
        f"{remaining_actor_missing[:20]}"
      )

  def load_state_dict(
    self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
  ) -> bool:
    # Distillation checkpoint path (student.* + teacher.*).
    if any(key.startswith("student.") for key in state_dict):
      resumed_training = super().load_state_dict(state_dict, strict=strict)
      self._load_teacher_actor_from_prefixed_state(state_dict)
      self.loaded_teacher = True
      self.teacher.eval()
      return resumed_training

    # Resume path for mixed-stage checkpoints (contains teacher.* directly).
    if any(key.startswith("teacher.") for key in state_dict):
      resumed_training = super().load_state_dict(state_dict, strict=strict)
      self.loaded_teacher = True
      self.teacher.eval()
      return resumed_training

    # Student-only checkpoint path: load actor/critic, keep teacher unloaded.
    actor_critic_state_dict = {
      key: value for key, value in state_dict.items() if not key.startswith("teacher.")
    }
    incompatible = nn.Module.load_state_dict(
      self, actor_critic_state_dict, strict=False
    )
    unresolved_non_teacher = [
      key for key in incompatible.missing_keys if not key.startswith("teacher.")
    ]
    if strict and len(unresolved_non_teacher) > 0:
      raise ValueError(
        "Failed to load non-teacher parameters from checkpoint: "
        f"{unresolved_non_teacher[:20]}"
      )
    self.loaded_teacher = False
    self.teacher.eval()
    return False
