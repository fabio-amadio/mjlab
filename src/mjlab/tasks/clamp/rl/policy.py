from __future__ import annotations

from functools import reduce
from typing import Mapping, Sequence

import torch
import torch.nn as nn
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.networks.mlp import resolve_nn_activation
from torch.distributions import Normal


class MotionEncoder(nn.Module):
  """Temporal encoder for stacked motion reference observations."""

  def __init__(
    self,
    input_dim_per_step: int,
    num_steps: int,
    latent_dim: int,
    activation: str = "elu",
    proj_channels: int = 60,
    conv_channels: tuple[int, ...] | list[int] = (40, 20),
    conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    conv_strides: tuple[int, ...] | list[int] = (2, 2),
  ) -> None:
    super().__init__()

    self.num_steps = int(num_steps)
    if self.num_steps <= 0:
      raise ValueError(f"`num_steps` must be positive, got {self.num_steps}.")
    input_dim_per_step = int(input_dim_per_step)
    if input_dim_per_step <= 0:
      raise ValueError(
        f"`input_dim_per_step` must be positive, got {input_dim_per_step}."
      )
    latent_dim = int(latent_dim)
    if latent_dim <= 0:
      raise ValueError(f"`latent_dim` must be positive, got {latent_dim}.")
    self.proj_dim = int(proj_channels)
    if self.proj_dim <= 0:
      raise ValueError(f"`proj_channels` must be positive, got {self.proj_dim}.")

    conv_channels = tuple(int(v) for v in conv_channels)
    conv_kernel_sizes = tuple(int(v) for v in conv_kernel_sizes)
    conv_strides = tuple(int(v) for v in conv_strides)
    if len(conv_channels) == 0:
      raise ValueError("`conv_channels` must contain at least one value.")
    if not (
      len(conv_channels) == len(conv_kernel_sizes) == len(conv_strides)
    ):
      raise ValueError(
        "Conv config lengths must match: "
        f"conv_layers={len(conv_channels)}, kernels={len(conv_kernel_sizes)}, strides={len(conv_strides)}."
      )
    if any(v <= 0 for v in conv_channels):
      raise ValueError(f"`conv_channels` must be positive, got {conv_channels}.")
    if any(v <= 0 for v in conv_kernel_sizes):
      raise ValueError(
        f"`conv_kernel_sizes` must be positive, got {conv_kernel_sizes}."
      )
    if any(v <= 0 for v in conv_strides):
      raise ValueError(f"`conv_strides` must be positive, got {conv_strides}.")

    self.proj = nn.Sequential(
      nn.Linear(input_dim_per_step, self.proj_dim),
      resolve_nn_activation(activation),
    )
    conv_layers: list[nn.Module] = []
    in_channels = self.proj_dim
    for out_channels, kernel, stride in zip(
      conv_channels, conv_kernel_sizes, conv_strides, strict=True
    ):
      conv_layers.append(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride)
      )
      conv_layers.append(resolve_nn_activation(activation))
      in_channels = out_channels
    self.conv = nn.Sequential(*conv_layers)
    self.flatten = nn.Flatten()

    conv_out_length = self.num_steps
    for kernel, stride in zip(conv_kernel_sizes, conv_strides, strict=True):
      conv_out_length = (conv_out_length - kernel) // stride + 1
      if conv_out_length <= 0:
        raise ValueError(
          "Invalid temporal conv config for given `num_steps`. "
          f"num_steps={self.num_steps}, conv_kernel_sizes={conv_kernel_sizes}, conv_strides={conv_strides}"
        )
    conv_out_dim = int(conv_channels[-1] * conv_out_length)

    self.out = nn.Linear(conv_out_dim, latent_dim)

  def forward(self, motion_obs: torch.Tensor) -> torch.Tensor:
    batch_size = motion_obs.shape[0]
    step_obs = motion_obs.reshape(batch_size, self.num_steps, -1)
    projected = self.proj(step_obs.reshape(batch_size * self.num_steps, -1))
    projected = projected.reshape(batch_size, self.num_steps, -1).permute(0, 2, 1)
    temporal = self.flatten(self.conv(projected))
    return self.out(temporal)


class ClampActorCriticMimic(nn.Module):
  """CLAMP teacher actor-critic with temporal motion encoder."""

  is_recurrent = False
  _VALID_NOISE_STD_TYPES = ("scalar", "log")

  def __init__(
    self,
    obs,
    obs_groups,
    num_actions,
    actor_obs_normalization: bool = False,
    critic_obs_normalization: bool = False,
    actor_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    critic_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    init_noise_std: float = 1.0,
    noise_std_type: str = "scalar",
    state_dependent_std: bool = False,
    motion_obs_dim: int = 1320,
    motion_steps: int = 20,
    motion_latent_dim: int = 128,
    motion_proj_channels: int = 60,
    motion_conv_channels: tuple[int, ...] | list[int] = (40, 20),
    motion_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    motion_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    layer_norm: bool = True,
    fix_action_std: bool = False,
    action_std: tuple[float, ...] | list[float] | None = None,
    share_obs_normalizer: bool | None = None,
    print_model_structure: bool = False,
    **kwargs,
  ):
    del kwargs
    super().__init__()

    self.obs_groups = obs_groups
    self.motion_obs_dim = int(motion_obs_dim)
    self.motion_steps = int(motion_steps)
    self.state_dependent_std = bool(state_dependent_std)
    self.noise_std_type = noise_std_type
    self.print_model_structure = bool(print_model_structure)

    if self.motion_obs_dim <= 0:
      raise ValueError(f"`motion_obs_dim` must be positive, got {self.motion_obs_dim}.")
    if self.motion_steps <= 0:
      raise ValueError(f"`motion_steps` must be positive, got {self.motion_steps}.")
    if self.motion_obs_dim % self.motion_steps != 0:
      raise ValueError(
        "Motion observation dimension must be divisible by motion_steps: "
        f"{self.motion_obs_dim} % {self.motion_steps} != 0."
      )

    num_actor_obs = self._infer_obs_dim(obs, obs_groups["policy"])
    num_critic_obs = self._infer_obs_dim(obs, obs_groups["critic"])

    if self.motion_obs_dim > num_actor_obs or self.motion_obs_dim > num_critic_obs:
      raise ValueError(
        "Configured motion_obs_dim is larger than actor/critic input dimensions: "
        f"motion_obs_dim={self.motion_obs_dim}, actor_obs={num_actor_obs}, critic_obs={num_critic_obs}."
      )

    self.single_motion_obs_dim = self.motion_obs_dim // self.motion_steps

    # Actor branch.
    self.actor_motion_encoder = MotionEncoder(
      input_dim_per_step=self.single_motion_obs_dim,
      num_steps=self.motion_steps,
      latent_dim=motion_latent_dim,
      activation=activation,
      proj_channels=motion_proj_channels,
      conv_channels=motion_conv_channels,
      conv_kernel_sizes=motion_conv_kernel_sizes,
      conv_strides=motion_conv_strides,
    )
    self._maybe_print_structure("Actor Motion Encoder", self.actor_motion_encoder)
    actor_backbone_in = (
      num_actor_obs
      - self.motion_obs_dim
      + self.single_motion_obs_dim
      + motion_latent_dim
    )
    actor_out_dim: int | list[int]
    if self.state_dependent_std:
      actor_out_dim = [2, num_actions]
    else:
      actor_out_dim = num_actions
    self.actor = self._build_mlp(
      input_dim=actor_backbone_in,
      output_dim=actor_out_dim,
      hidden_dims=actor_hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )
    self.actor_obs_normalization = actor_obs_normalization
    if actor_obs_normalization:
      self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
    else:
      self.actor_obs_normalizer = nn.Identity()
    self._maybe_print_structure("Actor MLP", self.actor)

    # Critic branch.
    self.critic_motion_encoder = MotionEncoder(
      input_dim_per_step=self.single_motion_obs_dim,
      num_steps=self.motion_steps,
      latent_dim=motion_latent_dim,
      activation=activation,
      proj_channels=motion_proj_channels,
      conv_channels=motion_conv_channels,
      conv_kernel_sizes=motion_conv_kernel_sizes,
      conv_strides=motion_conv_strides,
    )
    self._maybe_print_structure("Critic Motion Encoder", self.critic_motion_encoder)
    critic_backbone_in = (
      num_critic_obs
      - self.motion_obs_dim
      + self.single_motion_obs_dim
      + motion_latent_dim
    )
    self.critic = self._build_mlp(
      input_dim=critic_backbone_in,
      output_dim=1,
      hidden_dims=critic_hidden_dims,
      activation=activation,
      layer_norm=layer_norm,
    )
    self.critic_obs_normalization = critic_obs_normalization
    if critic_obs_normalization:
      if share_obs_normalizer is None:
        share_obs_normalizer = (
          actor_obs_normalization
          and num_actor_obs == num_critic_obs
          and tuple(obs_groups["policy"]) == tuple(obs_groups["critic"])
        )
      if share_obs_normalizer:
        if not actor_obs_normalization:
          raise ValueError(
            "`share_obs_normalizer=True` requires `actor_obs_normalization=True`."
          )
        if num_actor_obs != num_critic_obs:
          raise ValueError(
            "`share_obs_normalizer=True` requires same actor/critic obs dims: "
            f"{num_actor_obs} vs {num_critic_obs}."
          )
        self.critic_obs_normalizer = self.actor_obs_normalizer
      else:
        self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
    else:
      self.critic_obs_normalizer = nn.Identity()
    self._maybe_print_structure("Critic MLP", self.critic)

    # Action noise.
    if self.state_dependent_std:
      last_linear = self._last_linear(self.actor)
      torch.nn.init.zeros_(last_linear.weight[num_actions:])
      if self.noise_std_type == "scalar":
        torch.nn.init.constant_(last_linear.bias[num_actions:], init_noise_std)
      elif self.noise_std_type == "log":
        torch.nn.init.constant_(
          last_linear.bias[num_actions:],
          torch.log(torch.tensor(init_noise_std + 1e-7)),
        )
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. "
          f"Expected one of {self._VALID_NOISE_STD_TYPES}."
        )
    else:
      if self.noise_std_type == "scalar":
        if fix_action_std:
          if action_std is None:
            std_init = init_noise_std * torch.ones(num_actions)
          else:
            if len(action_std) != num_actions:
              raise ValueError(
                "`action_std` length must match num_actions when fix_action_std=True: "
                f"{len(action_std)} vs {num_actions}."
              )
            std_init = torch.tensor(action_std, dtype=torch.float32)
          self.std = nn.Parameter(std_init, requires_grad=False)
        else:
          self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
      elif self.noise_std_type == "log":
        if fix_action_std:
          if action_std is None:
            log_std_init = torch.log(init_noise_std * torch.ones(num_actions))
          else:
            if len(action_std) != num_actions:
              raise ValueError(
                "`action_std` length must match num_actions when fix_action_std=True: "
                f"{len(action_std)} vs {num_actions}."
              )
            log_std_init = torch.log(torch.tensor(action_std, dtype=torch.float32))
          self.log_std = nn.Parameter(log_std_init, requires_grad=False)
        else:
          self.log_std = nn.Parameter(
            torch.log(init_noise_std * torch.ones(num_actions))
          )
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. "
          f"Expected one of {self._VALID_NOISE_STD_TYPES}."
        )

    self.distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  @staticmethod
  def _build_mlp(
    input_dim: int,
    output_dim: int | list[int] | tuple[int, ...],
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    layer_norm: bool,
  ) -> nn.Sequential:
    """Builds an MLP with optional LayerNorm before the final hidden activation."""
    if len(hidden_dims) == 0:
      raise ValueError("`hidden_dims` must contain at least one element.")

    def new_activation() -> nn.Module:
      return resolve_nn_activation(activation)

    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_idx, hidden_dim in enumerate(hidden_dims):
      layers.append(nn.Linear(in_dim, hidden_dim))
      if layer_norm and len(hidden_dims) > 1 and hidden_idx == len(hidden_dims) - 1:
        layers.append(nn.LayerNorm(hidden_dim))
      layers.append(new_activation())
      in_dim = hidden_dim

    if isinstance(output_dim, int):
      layers.append(nn.Linear(in_dim, output_dim))
    else:
      total_out = reduce(lambda x, y: x * y, output_dim)
      layers.append(nn.Linear(in_dim, total_out))
      layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

    return nn.Sequential(*layers)

  @staticmethod
  def _last_linear(mlp: nn.Sequential) -> nn.Linear:
    for module in reversed(mlp):
      if isinstance(module, nn.Linear):
        return module
    raise RuntimeError("MLP has no Linear layer.")

  @staticmethod
  def _infer_obs_dim(
    obs: Mapping[str, torch.Tensor], group_names: Sequence[str]
  ) -> int:
    """Returns the concatenated flat dimension for the given observation groups."""
    total = 0
    for group_name in group_names:
      assert len(obs[group_name].shape) == 2, "Only 1D observations are supported."
      total += int(obs[group_name].shape[-1])
    return total

  def _maybe_print_structure(self, name: str, module: nn.Module) -> None:
    if self.print_model_structure:
      print(f"{name}: {module}")

  def reset(self, dones=None):
    del dones
    pass

  def forward(self):
    raise NotImplementedError

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

  def _split_motion_obs(
    self, obs_flat: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    motion_obs = obs_flat[:, : self.motion_obs_dim]
    remainder = obs_flat[:, self.motion_obs_dim :]
    return motion_obs, remainder

  def _encode_with_context(
    self, obs_flat: torch.Tensor, encoder: MotionEncoder
  ) -> torch.Tensor:
    """Encodes motion prefix and appends first-step motion for local context."""
    motion_obs, remainder = self._split_motion_obs(obs_flat)
    motion_latent = encoder(motion_obs)
    first_step_motion = motion_obs[:, : self.single_motion_obs_dim]
    return torch.cat((remainder, first_step_motion, motion_latent), dim=-1)

  def update_distribution(self, obs_flat: torch.Tensor):
    actor_input = self._encode_with_context(obs_flat, self.actor_motion_encoder)
    if self.state_dependent_std:
      mean_and_std = self.actor(actor_input)
      if self.noise_std_type == "scalar":
        mean, std = torch.unbind(mean_and_std, dim=-2)
      elif self.noise_std_type == "log":
        mean, log_std = torch.unbind(mean_and_std, dim=-2)
        std = torch.exp(log_std)
      else:
        raise ValueError(
          f"Unknown standard deviation type: {self.noise_std_type}. "
          f"Expected one of {self._VALID_NOISE_STD_TYPES}."
        )
    else:
      mean = self.actor(actor_input)
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

  def act(self, obs, **kwargs):
    del kwargs
    actor_obs = self.get_actor_obs(obs)
    actor_obs = self.actor_obs_normalizer(actor_obs)
    self.update_distribution(actor_obs)
    assert self.distribution is not None
    return self.distribution.sample()

  def act_inference(self, obs):
    actor_obs = self.get_actor_obs(obs)
    actor_obs = self.actor_obs_normalizer(actor_obs)
    actor_input = self._encode_with_context(actor_obs, self.actor_motion_encoder)
    return self.actor(actor_input)

  def evaluate(self, obs, **kwargs):
    del kwargs
    critic_obs = self.get_critic_obs(obs)
    critic_obs = self.critic_obs_normalizer(critic_obs)
    critic_input = self._encode_with_context(critic_obs, self.critic_motion_encoder)
    return self.critic(critic_input)

  def get_actor_obs(self, obs):
    return self._concat_obs_groups(obs, "policy")

  def get_critic_obs(self, obs):
    return self._concat_obs_groups(obs, "critic")

  def _concat_obs_groups(self, obs: Mapping[str, torch.Tensor], group_key: str):
    obs_list = [obs[group_name] for group_name in self.obs_groups[group_key]]
    return torch.cat(obs_list, dim=-1)

  def get_actions_log_prob(self, actions):
    assert self.distribution is not None
    return self.distribution.log_prob(actions).sum(dim=-1)

  def update_normalization(self, obs):
    if self.actor_obs_normalization:
      actor_obs = self.get_actor_obs(obs)
      self.actor_obs_normalizer.update(actor_obs)
    if self.critic_obs_normalization:
      critic_obs = self.get_critic_obs(obs)
      self.critic_obs_normalizer.update(critic_obs)
