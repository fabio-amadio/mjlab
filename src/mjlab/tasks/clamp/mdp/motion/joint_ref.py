"""Single-step joint-reference motion command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .representations import joint_ref_representation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class JointRefMotionCommand(MotionCommand):
  """Tracking-style command representation: [joint_pos_ref, joint_vel_ref]."""

  cfg: JointRefMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "default":
      return joint_ref_representation(self.joint_pos, self.joint_vel)
    return super().get_command_representation(representation_name)


@dataclass(kw_only=True)
class JointRefMotionCommandCfg(MotionCommandCfg):
  """Configuration for single-step joint-reference motion commands."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return JointRefMotionCommand(self, env)
