from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _body_indices_by_name(
  all_body_names: tuple[str, ...], selected_body_names: tuple[str, ...]
) -> list[int]:
  return [all_body_names.index(name) for name in selected_body_names]


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )

  return pos_b.view(env.num_envs, -1)


def robot_anchor_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.robot_anchor_pos_w[:, 2:3]


def robot_anchor_pos(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.robot_anchor_pos_w


def robot_anchor_rpy(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  roll, pitch, yaw = euler_xyz_from_quat(command.robot_anchor_quat_w)
  return torch.stack((roll, pitch, yaw), dim=-1)


def robot_key_body_pos_b(
  env: ManagerBasedRlEnv, command_name: str, key_body_names: tuple[str, ...]
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  key_body_indices = _body_indices_by_name(command.cfg.body_names, key_body_names)

  key_body_pos_w = command.robot_body_pos_w[:, key_body_indices, :]
  robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :]
  robot_anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(
    -1, len(key_body_indices), -1
  )
  key_body_pos_b = quat_apply_inverse(robot_anchor_quat_w, key_body_pos_w - robot_anchor_pos_w)
  return key_body_pos_b.reshape(env.num_envs, -1)


def feet_contact_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)

