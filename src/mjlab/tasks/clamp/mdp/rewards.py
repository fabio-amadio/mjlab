from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_tracking_joint_dof(
  env: ManagerBasedRlEnv,
  command_name: str,
  pos_scale: float = 0.15,
  dof_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  joint_diff = command.joint_pos - command.robot_joint_pos
  if dof_weights is not None:
    if len(dof_weights) != joint_diff.shape[1]:
      raise ValueError(
        "dof_weights length must match joint dimension: "
        f"weights={len(dof_weights)} joints={joint_diff.shape[1]}"
      )
    weights = torch.tensor(dof_weights, device=joint_diff.device, dtype=joint_diff.dtype)
  else:
    weights = torch.ones(joint_diff.shape[1], device=joint_diff.device, dtype=joint_diff.dtype)
  error = torch.sum(weights * torch.square(joint_diff), dim=-1)
  return torch.exp(-pos_scale * error)


def motion_tracking_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  vel_scale: float = 0.01,
  dof_weights: tuple[float, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  vel_diff = command.joint_vel - command.robot_joint_vel
  if dof_weights is not None:
    if len(dof_weights) != vel_diff.shape[1]:
      raise ValueError(
        "dof_weights length must match joint dimension: "
        f"weights={len(dof_weights)} joints={vel_diff.shape[1]}"
      )
    weights = torch.tensor(dof_weights, device=vel_diff.device, dtype=vel_diff.dtype)
  else:
    weights = torch.ones(vel_diff.shape[1], device=vel_diff.device, dtype=vel_diff.dtype)
  error = torch.sum(weights * torch.square(vel_diff), dim=-1)
  return torch.exp(-vel_scale * error)


def motion_tracking_root_pose(
  env: ManagerBasedRlEnv,
  command_name: str,
  root_pose_scale: float = 5.0,
  in_world_frame: bool = False,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if in_world_frame:
    root_pos_diff = command.anchor_pos_w - command.robot_anchor_pos_w
  else:
    root_pos_diff = command.anchor_pos_w[:, 2:3] - command.robot_anchor_pos_w[:, 2:3]
  root_pos_err = torch.sum(root_pos_diff * root_pos_diff, dim=-1)
  root_rot_err = (
    quat_error_magnitude(command.robot_anchor_quat_w, command.anchor_quat_w) ** 2
  )
  return torch.exp(-root_pose_scale * (root_pos_err + 0.1 * root_rot_err))


def motion_tracking_root_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
  root_vel_scale: float = 1.0,
  in_world_frame: bool = False,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if in_world_frame:
    root_vel_diff = command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w
    root_ang_vel_diff = command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w
  else:
    local_ref_root_vel = quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)
    local_ref_root_ang_vel = quat_apply_inverse(
      command.anchor_quat_w, command.anchor_ang_vel_w
    )
    local_robot_root_vel = quat_apply_inverse(
      command.robot_anchor_quat_w, command.robot_anchor_lin_vel_w
    )
    local_robot_root_ang_vel = quat_apply_inverse(
      command.robot_anchor_quat_w, command.robot_anchor_ang_vel_w
    )
    root_vel_diff = local_ref_root_vel - local_robot_root_vel
    root_ang_vel_diff = local_ref_root_ang_vel - local_robot_root_ang_vel
  root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
  root_ang_vel_err = torch.sum(root_ang_vel_diff * root_ang_vel_diff, dim=-1)
  return torch.exp(-root_vel_scale * (root_vel_err + 0.5 * root_ang_vel_err))


def motion_tracking_keybody_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  key_body_names: tuple[str, ...],
  key_body_pos_scale: float = 10.0,
  in_world_frame: bool = False,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, key_body_names)
  if len(body_indexes) == 0:
    return torch.zeros(env.num_envs, device=env.device)

  key_body_pos = command.robot_body_pos_w[:, body_indexes] - command.robot_anchor_pos_w[:, None, :]
  tar_key_body_pos = command.body_pos_w[:, body_indexes] - command.anchor_pos_w[:, None, :]

  if not in_world_frame:
    robot_yaw = yaw_quat(command.robot_anchor_quat_w)
    robot_yaw = robot_yaw[:, None, :].expand(-1, len(body_indexes), -1)
    ref_yaw = yaw_quat(command.anchor_quat_w)
    ref_yaw = ref_yaw[:, None, :].expand(-1, len(body_indexes), -1)
    key_body_pos = quat_apply_inverse(robot_yaw, key_body_pos)
    tar_key_body_pos = quat_apply_inverse(ref_yaw, tar_key_body_pos)

  key_body_pos_diff = key_body_pos - tar_key_body_pos
  key_body_pos_err = torch.sum(key_body_pos_diff * key_body_pos_diff, dim=-1)
  key_body_pos_err = torch.sum(key_body_pos_err, dim=-1)
  return torch.exp(-key_body_pos_scale * key_body_pos_err)
