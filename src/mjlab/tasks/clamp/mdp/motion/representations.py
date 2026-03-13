"""Tensor-serialization helpers for CLAMP motion command representations."""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

from .library import MotionFrameBatch


def joint_ref_representation(
  joint_pos: torch.Tensor, joint_vel: torch.Tensor
) -> torch.Tensor:
  """Serialize current joint reference state as [joint_pos, joint_vel]."""
  return torch.cat((joint_pos, joint_vel), dim=-1)


def future_joint_ref_representation(frames: MotionFrameBatch) -> torch.Tensor:
  """Serialize future joint references as [joint_pos, joint_vel]."""
  return torch.cat((frames.joint_pos, frames.joint_vel), dim=-1)


def future_joint_ref_anchor_representation(frames: MotionFrameBatch) -> torch.Tensor:
  """Serialize future joint references with anchor motion terms."""
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
