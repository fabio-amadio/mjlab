from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import euler_xyz_from_quat
from mjlab.utils.lab_api.math import quat_apply_inverse

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_root_height_diff(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.robot_anchor_pos_w[:, 2] - command.anchor_pos_w[:, 2]) > threshold
  )


def bad_roll_pitch(
  env: ManagerBasedRlEnv,
  roll_threshold: float,
  pitch_threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  roll, pitch, _ = euler_xyz_from_quat(asset.data.root_link_quat_w)
  roll_cut = torch.abs(roll) > roll_threshold
  pitch_cut = torch.abs(pitch) > pitch_threshold
  return roll_cut | pitch_cut


def root_lin_vel_too_large(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.norm(asset.data.root_link_lin_vel_w, dim=-1) > threshold


def motion_end(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  if command._uses_motion_library:
    assert command.motion_lib is not None
    motion_times = command.time_steps.to(torch.float32) * env.step_dt + command.motion_time_offsets
    motion_lengths = command.motion_lib.get_motion_length(command.motion_ids)
    return (motion_times + env.step_dt) >= motion_lengths
  assert command.motion is not None
  return (command.time_steps + 1) >= command.motion.time_step_total


def pose_termination(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...],
  in_world_frame: bool = False,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  if len(body_indexes) == 0:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  body_pos = command.robot_body_pos_w[:, body_indexes] - command.robot_anchor_pos_w[:, None, :]
  tar_body_pos = command.body_pos_w[:, body_indexes] - command.anchor_pos_w[:, None, :]

  if not in_world_frame:
    robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
      -1, len(body_indexes), -1
    )
    ref_anchor_quat = command.anchor_quat_w[:, None, :].expand(
      -1, len(body_indexes), -1
    )
    body_pos = quat_apply_inverse(robot_anchor_quat, body_pos)
    tar_body_pos = quat_apply_inverse(ref_anchor_quat, tar_body_pos)

  body_pos_diff = tar_body_pos - body_pos
  body_pos_dist = torch.sum(body_pos_diff * body_pos_diff, dim=-1)
  body_pos_dist = torch.max(body_pos_dist, dim=-1)[0]
  return body_pos_dist > threshold**2
