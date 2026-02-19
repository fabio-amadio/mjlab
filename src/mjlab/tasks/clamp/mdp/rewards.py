from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_anchor_height_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
  return torch.exp(-error / std**2)


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


def motion_tracking_root_vel_xy(
  env: ManagerBasedRlEnv,
  command_name: str,
  in_world_frame: bool = False,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if in_world_frame:
    root_vel_diff = command.anchor_lin_vel_w[:, :2] - command.robot_anchor_lin_vel_w[:, :2]
  else:
    local_ref_root_vel = quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)
    local_robot_root_vel = quat_apply_inverse(
      command.robot_anchor_quat_w, command.robot_anchor_lin_vel_w
    )
    root_vel_diff = local_ref_root_vel[:, :2] - local_robot_root_vel[:, :2]
  root_vel_err = torch.sum(root_vel_diff * root_vel_diff, dim=-1)
  return torch.exp(-root_vel_err)


def motion_tracking_root_ang_vel_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  in_world_frame: bool = False,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if in_world_frame:
    root_ang_vel_diff = command.anchor_ang_vel_w[:, 2] - command.robot_anchor_ang_vel_w[:, 2]
  else:
    local_ref_root_ang_vel = quat_apply_inverse(
      command.anchor_quat_w, command.anchor_ang_vel_w
    )
    local_robot_root_ang_vel = quat_apply_inverse(
      command.robot_anchor_quat_w, command.robot_anchor_ang_vel_w
    )
    root_ang_vel_diff = local_ref_root_ang_vel[:, 2] - local_robot_root_ang_vel[:, 2]
  root_ang_vel_err = root_ang_vel_diff * root_ang_vel_diff
  return torch.exp(-root_ang_vel_err)


def motion_tracking_root_height(
  env: ManagerBasedRlEnv, command_name: str, height_scale: float = 5.0
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  height_diff = command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
  height_err = height_diff * height_diff
  return torch.exp(-height_scale * height_err)


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


def motion_tracking_task_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  task_body_names: tuple[str, ...],
  task_body_pos_scale: float = 10.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, task_body_names)
  if len(body_indexes) == 0:
    return torch.zeros(env.num_envs, device=env.device)

  task_body_pos = command.robot_body_pos_w[:, body_indexes] - command.robot_anchor_pos_w[:, None, :]
  tar_body_pos = command.body_pos_w[:, body_indexes] - command.anchor_pos_w[:, None, :]

  robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
    -1, len(body_indexes), -1
  )
  ref_anchor_quat = command.anchor_quat_w[:, None, :].expand(-1, len(body_indexes), -1)
  task_body_pos = quat_apply_inverse(robot_anchor_quat, task_body_pos)
  tar_body_pos = quat_apply_inverse(ref_anchor_quat, tar_body_pos)

  task_body_pos_diff = task_body_pos - tar_body_pos
  task_body_pos_err = torch.sum(task_body_pos_diff * task_body_pos_diff, dim=-1)
  task_body_pos_err = torch.sum(task_body_pos_err, dim=-1)
  return torch.exp(-task_body_pos_scale * task_body_pos_err)


def motion_tracking_task_body_rot(
  env: ManagerBasedRlEnv,
  command_name: str,
  task_body_names: tuple[str, ...],
  task_body_rot_scale: float = 5.0,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, task_body_names)
  if len(body_indexes) == 0:
    return torch.zeros(env.num_envs, device=env.device)

  task_body_rot = command.robot_body_quat_w[:, body_indexes]
  ref_task_body_rot = command.body_quat_w[:, body_indexes]

  root_inv_rot = quat_inv(command.robot_anchor_quat_w)[:, None, :].expand(
    -1, len(body_indexes), -1
  )
  ref_root_inv_rot = quat_inv(command.anchor_quat_w)[:, None, :].expand(
    -1, len(body_indexes), -1
  )
  local_task_rot = quat_mul(root_inv_rot, task_body_rot)
  local_ref_task_rot = quat_mul(ref_root_inv_rot, ref_task_body_rot)

  rot_err = quat_error_magnitude(local_task_rot, local_ref_task_rot) ** 2
  rot_err = rot_err.sum(dim=1)
  return torch.exp(-task_body_rot_scale * rot_err)


def motion_local_anchor_linear_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, xy_only: bool = False
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_lin_vel_b = quat_apply_inverse(command.anchor_quat_w, command.anchor_lin_vel_w)
  robot_lin_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot_anchor_lin_vel_w
  )
  if xy_only:
    ref_lin_vel_b = ref_lin_vel_b[:, :2]
    robot_lin_vel_b = robot_lin_vel_b[:, :2]
  error = torch.sum(torch.square(ref_lin_vel_b - robot_lin_vel_b), dim=-1)
  return torch.exp(-error / std**2)


def motion_local_anchor_angular_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, yaw_only: bool = False
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  ref_ang_vel_b = quat_apply_inverse(command.anchor_quat_w, command.anchor_ang_vel_w)
  robot_ang_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot_anchor_ang_vel_w
  )
  if yaw_only:
    ref_ang_vel_b = ref_ang_vel_b[:, 2:3]
    robot_ang_vel_b = robot_ang_vel_b[:, 2:3]
  error = torch.sum(torch.square(ref_ang_vel_b - robot_ang_vel_b), dim=-1)
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
  return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(torch.square(command.joint_vel - command.robot_joint_vel), dim=-1)
  return torch.exp(-error / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Track body linear velocity in anchor-local coordinates.

  The reference and robot body velocities are each expressed in their own anchor
  body frame before comparison. This mirrors the local-frame philosophy used for
  CLAMP (`in_world_frame=False` equivalent).
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  ref_anchor_quat = command.anchor_quat_w[:, None, :].expand(
    -1, command.body_lin_vel_w.shape[1], -1
  )
  robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
    -1, command.robot_body_lin_vel_w.shape[1], -1
  )

  ref_body_lin_vel_b = quat_apply_inverse(ref_anchor_quat, command.body_lin_vel_w)
  robot_body_lin_vel_b = quat_apply_inverse(
    robot_anchor_quat, command.robot_body_lin_vel_w
  )

  error = torch.sum(
    torch.square(
      ref_body_lin_vel_b[:, body_indexes] - robot_body_lin_vel_b[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_local_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """Track body angular velocity in anchor-local coordinates."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)

  ref_anchor_quat = command.anchor_quat_w[:, None, :].expand(
    -1, command.body_ang_vel_w.shape[1], -1
  )
  robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
    -1, command.robot_body_ang_vel_w.shape[1], -1
  )

  ref_body_ang_vel_b = quat_apply_inverse(ref_anchor_quat, command.body_ang_vel_w)
  robot_body_ang_vel_b = quat_apply_inverse(
    robot_anchor_quat, command.robot_body_ang_vel_w
  )

  error = torch.sum(
    torch.square(
      ref_body_ang_vel_b[:, body_indexes] - robot_body_ang_vel_b[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  foot_body_names: tuple[str, ...],
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]
  asset: Entity = env.scene[command.cfg.entity_name]

  body_ids, _ = asset.find_bodies(foot_body_names)
  if len(body_ids) == 0:
    return torch.zeros(env.num_envs, device=env.device)

  assert sensor.data.force is not None
  in_contact = sensor.data.force[..., 2] > 5.0
  foot_speed_norm = torch.norm(asset.data.body_link_lin_vel_w[:, body_ids, :2], dim=-1)
  rew = torch.sqrt(torch.clamp(foot_speed_norm, min=0.0))
  rew *= in_contact.float()
  return torch.sum(rew, dim=1)


def feet_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  z_forces = sensor.data.force[..., 2]
  rew = torch.norm(z_forces, dim=-1)
  rew = torch.where(
    rew < max_contact_force,
    torch.zeros_like(rew),
    rew - max_contact_force,
  )
  return rew


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  ratio: float = 4.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_xy = torch.norm(sensor.data.force[..., :2], dim=-1)
  force_z = torch.abs(sensor.data.force[..., 2])
  rew = torch.any(force_xy > ratio * force_z, dim=1)
  return rew.float()


def dof_torque_limits(
  env: ManagerBasedRlEnv,
  soft_torque_limit: float = 0.95,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  torques = torch.abs(asset.data.actuator_force)
  torque_limits = torch.full_like(torques, float("inf"))
  for act in asset.actuators:
    ctrl_ids = act.ctrl_ids
    force_limit = getattr(act, "force_limit", None)
    if isinstance(force_limit, torch.Tensor):
      limit = force_limit
    else:
      effort_limit = getattr(act.cfg, "effort_limit", None)
      limit_value = float("inf") if effort_limit is None else float(effort_limit)
      limit = torch.full(
        (env.num_envs, ctrl_ids.numel()),
        limit_value,
        device=env.device,
        dtype=torques.dtype,
      )
    torque_limits[:, ctrl_ids] = limit
  ratio = torques / torch.clamp(torque_limits, min=1.0e-6)
  out_of_limits = torch.clamp(ratio - soft_torque_limit, min=0.0)
  return torch.sum(out_of_limits, dim=1)


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  target_air_time: float = 0.5,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  first_contact = sensor.compute_first_contact(dt=env.step_dt)
  assert sensor_data.last_air_time is not None
  air_time = (sensor_data.last_air_time - target_air_time) * first_contact.float()
  air_time = air_time.clamp(max=0.0)
  rew_airtime = air_time.sum(dim=1)
  rew_airtime *= torch.norm(command.anchor_lin_vel_w[:, :2], dim=1) > 0.05
  return rew_airtime


def ang_vel_xy(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
