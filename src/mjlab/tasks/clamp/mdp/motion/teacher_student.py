"""Synchronized student/teacher motion command definitions for distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .joint_ref import JointRefMotionCommand
from .representations import (
  future_joint_ref_anchor_representation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TeacherStudentMotionCommand(JointRefMotionCommand):
  """Joint-ref student command with synchronized future teacher command view."""

  cfg: TeacherStudentMotionCommandCfg

  @property
  def command_representation_names(self) -> tuple[str, ...]:
    return ("default", "teacher")

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "teacher":
      frames = self.query_motion_frames(self.cfg.future_sampling_step_offsets)
      return future_joint_ref_anchor_representation(frames).reshape(self.num_envs, -1)
    return super().get_command_representation(representation_name)

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    return self.cfg.future_sampling_step_offsets


@dataclass(kw_only=True)
class TeacherStudentMotionCommandCfg(MotionCommandCfg):
  """Configuration for synchronized joint-ref student and teacher command views."""

  future_sampling_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.future_sampling_step_offsets) == 0:
      raise ValueError(
        "`future_sampling_step_offsets` must be non-empty for TeacherStudentMotionCommandCfg."
      )
    return TeacherStudentMotionCommand(self, env)
