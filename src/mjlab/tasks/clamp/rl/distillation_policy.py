from __future__ import annotations

from typing import Mapping, Sequence

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization
from torch.distributions import Normal

from mjlab.tasks.clamp.rl.policy import ClampActorCriticMimic


class ClampStudentTeacherDistill(nn.Module):
  """Student-teacher distillation policy using CLAMP teacher actor + MLP student."""

  is_recurrent = False
  _VALID_NOISE_STD_TYPES = ("scalar", "log")

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions: int,
    student_obs_normalization: bool = True,
    teacher_obs_normalization: bool = True,
    student_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
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
    del kwargs
    super().__init__()

    self.loaded_teacher = False
    self.obs_groups = obs_groups
    self.noise_std_type = noise_std_type

    num_student_obs = self._infer_obs_dim(obs, obs_groups["policy"])
    self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
    self.student_obs_normalization = student_obs_normalization
    if student_obs_normalization:
      self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
    else:
      self.student_obs_normalizer = nn.Identity()

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

    if self.noise_std_type == "scalar":
      self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
    elif self.noise_std_type == "log":
      self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. "
        f"Expected one of {self._VALID_NOISE_STD_TYPES}."
      )

    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  @staticmethod
  def _infer_obs_dim(
    obs: Mapping[str, torch.Tensor], group_names: Sequence[str]
  ) -> int:
    total = 0
    for group_name in group_names:
      assert len(obs[group_name].shape) == 2, "Only 1D observations are supported."
      total += int(obs[group_name].shape[-1])
    return total

  @property
  def action_mean(self):
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def action_std(self):
    assert self.distribution is not None
    return self.distribution.stddev

  @property
  def entropy(self):
    assert self.distribution is not None
    return self.distribution.entropy().sum(dim=-1)

  def reset(self, dones=None, hidden_states=None):
    del dones, hidden_states
    pass

  def forward(self):
    raise NotImplementedError

  def get_hidden_states(self):
    return None

  def detach_hidden_states(self, dones=None):
    del dones
    pass

  def _get_student_obs(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
    obs_list = [obs[group_name] for group_name in self.obs_groups["policy"]]
    return torch.cat(obs_list, dim=-1)

  def update_distribution(self, student_obs: torch.Tensor) -> None:
    mean = self.student(student_obs)
    if self.noise_std_type == "scalar":
      std = self.std.expand_as(mean)
    elif self.noise_std_type == "log":
      std = torch.exp(self.log_std).expand_as(mean)
    else:
      raise ValueError(
        f"Unknown standard deviation type: {self.noise_std_type}. "
        f"Expected one of {self._VALID_NOISE_STD_TYPES}."
      )
    self.distribution = Normal(mean, std)

  def act(self, obs):
    student_obs = self._get_student_obs(obs)
    student_obs = self.student_obs_normalizer(student_obs)
    self.update_distribution(student_obs)
    assert self.distribution is not None
    return self.distribution.sample()

  def act_inference(self, obs):
    student_obs = self._get_student_obs(obs)
    student_obs = self.student_obs_normalizer(student_obs)
    return self.student(student_obs)

  def evaluate(self, obs):
    with torch.no_grad():
      return self.teacher.act_inference(obs)

  def train(self, mode=True):
    super().train(mode)
    self.teacher.eval()

  def update_normalization(self, obs):
    if self.student_obs_normalization:
      student_obs = self._get_student_obs(obs)
      self.student_obs_normalizer.update(student_obs)

  def load_state_dict(self, state_dict, strict=True):
    if any(key.startswith("student.") for key in state_dict.keys()):
      super().load_state_dict(state_dict, strict=strict)
      self.loaded_teacher = True
      self.teacher.eval()
      return True

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
      if key.startswith("actor.") or key.startswith("actor_motion_encoder."):
        actor_state_dict[key] = value
      elif key.startswith("actor_obs_normalizer."):
        actor_state_dict[key] = value
      elif key in ("std", "log_std"):
        actor_state_dict[key] = value

    if len(actor_state_dict) == 0:
      raise ValueError(
        "state_dict does not contain distillation state (student.*) or CLAMP teacher actor parameters."
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

    self.loaded_teacher = True
    self.teacher.eval()
    return False
