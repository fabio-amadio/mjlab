from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .motion_command import MotionCommand, MotionCommandCfg
from .motion_library import MotionFrameBatch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class FutureMotionCommand(MotionCommand):
  """Base class for cached future-stack command representations."""

  cfg: FutureJointRefMotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command_cache: torch.Tensor | None = None

  def _compose_future_features(self, frames: MotionFrameBatch) -> torch.Tensor:
    raise NotImplementedError

  def _build_cached_command(self) -> torch.Tensor:
    frames = self.query_motion_frames(self.cfg.command_step_offsets)
    command = self._compose_future_features(frames)
    return command.reshape(self.num_envs, -1)

  def _refresh_command_cache(self) -> None:
    self._command_cache = self._build_cached_command()

  @property
  def command(self) -> torch.Tensor:
    if self._command_cache is None:
      self._refresh_command_cache()
    assert self._command_cache is not None
    return self._command_cache

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self._command_cache = None

  def _update_command(self):
    super()._update_command()
    self._refresh_command_cache()


class FutureJointRefMotionCommand(FutureMotionCommand):
  """Future stacked joint reference command representation."""

  def _compose_future_features(self, frames: MotionFrameBatch) -> torch.Tensor:
    return torch.cat((frames.joint_pos, frames.joint_vel), dim=-1)


class FutureJointRefAnchorMotionCommand(FutureMotionCommand):
  """Future stacked joint references with anchor motion terms."""

  def _compose_future_features(self, frames: MotionFrameBatch) -> torch.Tensor:
    anchor_lin_vel_b = quat_apply_inverse(frames.anchor_quat_w, frames.anchor_lin_vel_w)
    anchor_ang_vel_b = quat_apply_inverse(frames.anchor_quat_w, frames.anchor_ang_vel_w)
    anchor_lin_vel_xy = anchor_lin_vel_b[..., :2]
    anchor_ang_vel_z = anchor_ang_vel_b[..., 2:3]
    return torch.cat(
      [
        frames.joint_pos,
        frames.joint_vel,
        anchor_lin_vel_xy,
        anchor_ang_vel_z,
        frames.anchor_pos_w[..., 2:3],
      ],
      dim=-1,
    )


@dataclass(kw_only=True)
class FutureJointRefMotionCommandCfg(MotionCommandCfg):
  """Future stacked joint-ref command configuration."""

  command_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for FutureJointRefMotionCommandCfg."
      )
    return FutureJointRefMotionCommand(self, env)


@dataclass(kw_only=True)
class FutureJointRefAnchorMotionCommandCfg(FutureJointRefMotionCommandCfg):
  """Future stacked joint-ref+anchor-motion command configuration."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for FutureJointRefAnchorMotionCommandCfg."
      )
    return FutureJointRefAnchorMotionCommand(self, env)
