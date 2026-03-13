"""Future-stacked joint-reference command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .library import MotionFrameBatch
from .representations import (
  future_joint_ref_anchor_representation,
  future_joint_ref_representation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class FutureJointRefMotionCommand(MotionCommand):
  """Future stacked joint reference command representation."""

  cfg: FutureJointRefMotionCommandCfg

  def _compose_future_features(self, frames: MotionFrameBatch) -> torch.Tensor:
    return future_joint_ref_representation(frames)

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name != "default":
      return super().get_command_representation(representation_name)
    frames = self.query_motion_frames(self.cfg.command_step_offsets)
    command = self._compose_future_features(frames)
    return command.reshape(self.num_envs, -1)

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    return self.cfg.command_step_offsets


class FutureJointRefAnchorMotionCommand(FutureJointRefMotionCommand):
  """Future stacked joint references with anchor motion terms."""

  cfg: FutureJointRefAnchorMotionCommandCfg

  def _compose_future_features(self, frames: MotionFrameBatch) -> torch.Tensor:
    return future_joint_ref_anchor_representation(frames)


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
