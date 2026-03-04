from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .student_commands import (
  HandBaseMotionCommand,
  TeacherStudentMotionCommand,
  _rot6d_from_matrix,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _student_command_term(
  env: ManagerBasedRlEnv, command_name: str
) -> HandBaseMotionCommand | TeacherStudentMotionCommand:
  command = env.command_manager.get_term(command_name)
  if isinstance(command, (HandBaseMotionCommand, TeacherStudentMotionCommand)):
    return command
  raise TypeError(
    f"Command '{command_name}' is not a student hand-base command term. "
    f"Got: {type(command)}"
  )


def _student_command_tensor(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command_term = env.command_manager.get_term(command_name)
  if isinstance(command_term, TeacherStudentMotionCommand):
    return command_term.student_command
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def _current_hand_pose_in_anchor_frame(
  command: HandBaseMotionCommand | TeacherStudentMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
  left_idx = command.left_hand_body_index
  right_idx = command.right_hand_body_index
  left_robot_idx = int(command.robot_body_indexes[left_idx].item())
  right_robot_idx = int(command.robot_body_indexes[right_idx].item())

  hand_pos_w = torch.stack(
    [
      command.robot.data.body_link_pos_w[:, left_robot_idx],
      command.robot.data.body_link_pos_w[:, right_robot_idx],
    ],
    dim=1,
  )
  hand_quat_w = torch.stack(
    [
      command.robot.data.body_link_quat_w[:, left_robot_idx],
      command.robot.data.body_link_quat_w[:, right_robot_idx],
    ],
    dim=1,
  )

  anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand(-1, 2, -1)
  anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(-1, 2, -1)
  hand_pos_b, hand_quat_b = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    hand_pos_w,
    hand_quat_w,
  )
  return hand_pos_b, hand_quat_b


def hand_position_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _student_command_term(env, command_name)
  student_command = _student_command_tensor(env, command_name)
  desired_pos_b = student_command[:, :6].reshape(-1, 2, 3)
  actual_pos_b, _ = _current_hand_pose_in_anchor_frame(command)
  error = torch.sum(torch.square(desired_pos_b - actual_pos_b), dim=-1).mean(dim=-1)
  return torch.exp(-error / std**2)


def hand_orientation_tracking_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _student_command_term(env, command_name)
  student_command = _student_command_tensor(env, command_name)
  desired_rot6d_b = student_command[:, 6:18].reshape(-1, 2, 6)
  _, actual_quat_b = _current_hand_pose_in_anchor_frame(command)
  actual_rot6d_b = _rot6d_from_matrix(matrix_from_quat(actual_quat_b))
  error = torch.sum(torch.square(desired_rot6d_b - actual_rot6d_b), dim=-1).mean(
    dim=-1
  )
  return torch.exp(-error / std**2)


def track_base_linear_velocity_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _student_command_term(env, command_name)
  student_command = _student_command_tensor(env, command_name)
  desired_xy = student_command[:, 18:20]
  actual_lin_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w,
    command.robot_anchor_lin_vel_w,
  )
  xy_error = torch.sum(torch.square(desired_xy - actual_lin_vel_b[:, :2]), dim=1)
  z_error = torch.square(actual_lin_vel_b[:, 2])
  lin_vel_error = xy_error + z_error
  return torch.exp(-lin_vel_error / std**2)


def track_base_angular_velocity_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = _student_command_term(env, command_name)
  student_command = _student_command_tensor(env, command_name)
  desired_wz = student_command[:, 20]
  actual_ang_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w,
    command.robot_anchor_ang_vel_w,
  )
  z_error = torch.square(desired_wz - actual_ang_vel_b[:, 2])
  xy_error = torch.sum(torch.square(actual_ang_vel_b[:, :2]), dim=1)
  ang_vel_error = z_error + xy_error
  return torch.exp(-ang_vel_error / std**2)


def _command_activity_mask(
  student_command: torch.Tensor, command_threshold: float
) -> torch.Tensor:
  linear_norm = torch.norm(student_command[:, 18:20], dim=1)
  angular_norm = torch.abs(student_command[:, 20])
  total_command = linear_norm + angular_norm
  return (total_command > command_threshold).float()


def feet_slip_hand_base(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  student_command = _student_command_tensor(env, command_name)
  active = _command_activity_mask(student_command, command_threshold)

  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  vel_xy_norm_sq = torch.square(vel_xy_norm)
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active

  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing_hand_base(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force
  force_magnitude = torch.norm(forces, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)

  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras.setdefault("log", {})
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force

  if command_name is not None:
    student_command = _student_command_tensor(env, command_name)
    active = _command_activity_mask(student_command, command_threshold)
    cost = cost * active
  return cost


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)
