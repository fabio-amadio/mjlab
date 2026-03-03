from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .motion_command import MotionCommand, MotionCommandCfg
from .motion_library import MotionFrameBatch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HandBaseMotionCommand(MotionCommand):
  """Single-step student command: hand poses + base velocity command."""

  cfg: HandBaseMotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert isinstance(cfg, HandBaseMotionCommandCfg)
    self.cfg = cfg

    name_to_idx = {name: idx for idx, name in enumerate(self.cfg.body_names)}
    try:
      self._left_hand_idx = name_to_idx[self.cfg.left_hand_body_name]
      self._right_hand_idx = name_to_idx[self.cfg.right_hand_body_name]
    except KeyError as exc:
      raise ValueError(
        "Hand body names must be included in `body_names` for HandBaseMotionCommand. "
        f"Missing: {exc}"
      ) from exc

    self._command_cache: torch.Tensor | None = None

  def _build_command(self) -> torch.Tensor:
    # Current desired reference frame (single step).
    frames = self.query_motion_frames((0,))
    body_pos_w = frames.body_pos_w[:, 0]
    body_quat_w = frames.body_quat_w[:, 0]
    anchor_lin_vel_w = frames.anchor_lin_vel_w[:, 0]
    anchor_ang_vel_w = frames.anchor_ang_vel_w[:, 0]

    hand_pos_w = torch.stack(
      [body_pos_w[:, self._left_hand_idx], body_pos_w[:, self._right_hand_idx]], dim=1
    )
    hand_quat_w = torch.stack(
      [body_quat_w[:, self._left_hand_idx], body_quat_w[:, self._right_hand_idx]], dim=1
    )

    robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, 2, -1)
    robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, 2, -1)
    hand_pos_b, hand_quat_b = subtract_frame_transforms(
      robot_anchor_pos,
      robot_anchor_quat,
      hand_pos_w,
      hand_quat_w,
    )
    hand_rot6d_b = matrix_from_quat(hand_quat_b)[..., :2].reshape(self.num_envs, 2, 6)

    # Desired anchor velocity, expressed in current robot anchor frame.
    cmd_lin_vel_b = quat_apply_inverse(self.robot_anchor_quat_w, anchor_lin_vel_w)
    cmd_ang_vel_b = quat_apply_inverse(self.robot_anchor_quat_w, anchor_ang_vel_w)

    return torch.cat(
      [
        hand_pos_b.reshape(self.num_envs, -1),  # 2 x 3
        hand_rot6d_b.reshape(self.num_envs, -1),  # 2 x 6
        cmd_lin_vel_b[:, :2],  # vx, vy
        cmd_ang_vel_b[:, 2:3],  # wz
      ],
      dim=-1,
    )

  def _refresh_command_cache(self) -> None:
    self._command_cache = self._build_command()

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


class TeacherStudentMotionCommand(MotionCommand):
  """Shared motion command exposing both student and teacher command views."""

  cfg: TeacherStudentMotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert isinstance(cfg, TeacherStudentMotionCommandCfg)
    self.cfg = cfg

    name_to_idx = {name: idx for idx, name in enumerate(self.cfg.body_names)}
    try:
      self._left_hand_idx = name_to_idx[self.cfg.left_hand_body_name]
      self._right_hand_idx = name_to_idx[self.cfg.right_hand_body_name]
    except KeyError as exc:
      raise ValueError(
        "Hand body names must be included in `body_names` for TeacherStudentMotionCommand. "
        f"Missing: {exc}"
      ) from exc

    self._student_command_cache: torch.Tensor | None = None
    self._teacher_command_cache: torch.Tensor | None = None

  def _build_student_command(self) -> torch.Tensor:
    frames = self.query_motion_frames((0,))
    body_pos_w = frames.body_pos_w[:, 0]
    body_quat_w = frames.body_quat_w[:, 0]
    anchor_lin_vel_w = frames.anchor_lin_vel_w[:, 0]
    anchor_ang_vel_w = frames.anchor_ang_vel_w[:, 0]

    hand_pos_w = torch.stack(
      [body_pos_w[:, self._left_hand_idx], body_pos_w[:, self._right_hand_idx]], dim=1
    )
    hand_quat_w = torch.stack(
      [body_quat_w[:, self._left_hand_idx], body_quat_w[:, self._right_hand_idx]], dim=1
    )

    robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand(-1, 2, -1)
    robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand(-1, 2, -1)
    hand_pos_b, hand_quat_b = subtract_frame_transforms(
      robot_anchor_pos,
      robot_anchor_quat,
      hand_pos_w,
      hand_quat_w,
    )
    hand_rot6d_b = matrix_from_quat(hand_quat_b)[..., :2].reshape(self.num_envs, 2, 6)

    cmd_lin_vel_b = quat_apply_inverse(self.robot_anchor_quat_w, anchor_lin_vel_w)
    cmd_ang_vel_b = quat_apply_inverse(self.robot_anchor_quat_w, anchor_ang_vel_w)

    return torch.cat(
      [
        hand_pos_b.reshape(self.num_envs, -1),  # 2 x 3
        hand_rot6d_b.reshape(self.num_envs, -1),  # 2 x 6
        cmd_lin_vel_b[:, :2],  # vx, vy
        cmd_ang_vel_b[:, 2:3],  # wz
      ],
      dim=-1,
    )

  def _build_teacher_command(self) -> torch.Tensor:
    frames = self.query_motion_frames(self.cfg.command_step_offsets)
    return self._compose_teacher_future_features(frames).reshape(self.num_envs, -1)

  @staticmethod
  def _compose_teacher_future_features(frames: MotionFrameBatch) -> torch.Tensor:
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

  def _refresh_command_cache(self) -> None:
    self._student_command_cache = self._build_student_command()
    self._teacher_command_cache = self._build_teacher_command()

  @property
  def command(self) -> torch.Tensor:
    # Backward-compatible default command view: student command.
    return self.student_command

  @property
  def student_command(self) -> torch.Tensor:
    if self._student_command_cache is None:
      self._refresh_command_cache()
    assert self._student_command_cache is not None
    return self._student_command_cache

  @property
  def teacher_command(self) -> torch.Tensor:
    if self._teacher_command_cache is None:
      self._refresh_command_cache()
    assert self._teacher_command_cache is not None
    return self._teacher_command_cache

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self._student_command_cache = None
    self._teacher_command_cache = None

  def _update_command(self):
    super()._update_command()
    self._refresh_command_cache()


@dataclass(kw_only=True)
class HandBaseMotionCommandCfg(MotionCommandCfg):
  """Configuration for student hand+base command representation."""

  left_hand_body_name: str = ""
  right_hand_body_name: str = ""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if self.left_hand_body_name == "" or self.right_hand_body_name == "":
      raise ValueError(
        "`left_hand_body_name` and `right_hand_body_name` must be set for HandBaseMotionCommandCfg."
      )
    return HandBaseMotionCommand(self, env)


@dataclass(kw_only=True)
class TeacherStudentMotionCommandCfg(MotionCommandCfg):
  """Configuration for combined student and teacher motion command views."""

  left_hand_body_name: str = ""
  right_hand_body_name: str = ""
  command_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if self.left_hand_body_name == "" or self.right_hand_body_name == "":
      raise ValueError(
        "`left_hand_body_name` and `right_hand_body_name` must be set for TeacherStudentMotionCommandCfg."
      )
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for TeacherStudentMotionCommandCfg."
      )
    return TeacherStudentMotionCommand(self, env)
