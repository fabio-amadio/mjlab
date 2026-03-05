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
from .motion_command_hand_base import _debug_vis_hand_base_command, _rot6d_from_matrix
from .motion_library import MotionFrameBatch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class DualViewMotionCommand(MotionCommand):
  """Shared motion command exposing both student and teacher command views."""

  cfg: DualViewMotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert isinstance(cfg, DualViewMotionCommandCfg)
    self.cfg = cfg

    name_to_idx = {name: idx for idx, name in enumerate(self.cfg.body_names)}
    try:
      self._left_hand_idx = name_to_idx[self.cfg.left_hand_body_name]
      self._right_hand_idx = name_to_idx[self.cfg.right_hand_body_name]
    except KeyError as exc:
      raise ValueError(
        "Hand body names must be included in `body_names` for DualViewMotionCommand. "
        f"Missing: {exc}"
      ) from exc

    self._student_command_cache: torch.Tensor | None = None
    self._teacher_command_cache: torch.Tensor | None = None
    self._init_student_metrics()

  def _init_student_metrics(self) -> None:
    self.metrics["student_hand_pos_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["student_hand_ori_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["student_base_lin_vel_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["student_base_ang_vel_error"] = torch.zeros(
      self.num_envs, device=self.device
    )

  def _update_student_metrics(self) -> None:
    student_cmd = self.student_command
    desired_hand_pos_b = student_cmd[:, :6].reshape(self.num_envs, 2, 3)
    desired_hand_rot6d_b = student_cmd[:, 6:18].reshape(self.num_envs, 2, 6)
    desired_lin_vel_xy_b = student_cmd[:, 18:20]
    desired_ang_vel_z_b = student_cmd[:, 20]

    left_robot_idx = int(self.robot_body_indexes[self.left_hand_body_index].item())
    right_robot_idx = int(self.robot_body_indexes[self.right_hand_body_index].item())

    hand_pos_w = torch.stack(
      [
        self.robot.data.body_link_pos_w[:, left_robot_idx],
        self.robot.data.body_link_pos_w[:, right_robot_idx],
      ],
      dim=1,
    )
    hand_quat_w = torch.stack(
      [
        self.robot.data.body_link_quat_w[:, left_robot_idx],
        self.robot.data.body_link_quat_w[:, right_robot_idx],
      ],
      dim=1,
    )

    anchor_pos_w = self.robot_anchor_pos_w[:, None, :].expand(-1, 2, -1)
    anchor_quat_w = self.robot_anchor_quat_w[:, None, :].expand(-1, 2, -1)
    hand_pos_b, hand_quat_b = subtract_frame_transforms(
      anchor_pos_w,
      anchor_quat_w,
      hand_pos_w,
      hand_quat_w,
    )
    hand_rot6d_b = _rot6d_from_matrix(matrix_from_quat(hand_quat_b))

    actual_lin_vel_b = quat_apply_inverse(
      self.robot_anchor_quat_w, self.robot_anchor_lin_vel_w
    )
    actual_ang_vel_b = quat_apply_inverse(
      self.robot_anchor_quat_w, self.robot_anchor_ang_vel_w
    )

    self.metrics["student_hand_pos_error"] = torch.norm(
      desired_hand_pos_b - hand_pos_b, dim=-1
    ).mean(dim=-1)
    self.metrics["student_hand_ori_error"] = torch.norm(
      desired_hand_rot6d_b - hand_rot6d_b, dim=-1
    ).mean(dim=-1)
    self.metrics["student_base_lin_vel_error"] = torch.norm(
      desired_lin_vel_xy_b - actual_lin_vel_b[:, :2], dim=-1
    )
    self.metrics["student_base_ang_vel_error"] = torch.abs(
      desired_ang_vel_z_b - actual_ang_vel_b[:, 2]
    )

  def _build_student_command(self) -> torch.Tensor:
    frames = self.query_motion_frames((0,))
    body_pos_w = frames.body_pos_w[:, 0]
    body_quat_w = frames.body_quat_w[:, 0]
    anchor_pos_w = frames.anchor_pos_w[:, 0]
    anchor_quat_w = frames.anchor_quat_w[:, 0]
    anchor_lin_vel_w = frames.anchor_lin_vel_w[:, 0]
    anchor_ang_vel_w = frames.anchor_ang_vel_w[:, 0]

    hand_pos_w = torch.stack(
      [body_pos_w[:, self._left_hand_idx], body_pos_w[:, self._right_hand_idx]], dim=1
    )
    hand_quat_w = torch.stack(
      [body_quat_w[:, self._left_hand_idx], body_quat_w[:, self._right_hand_idx]], dim=1
    )

    anchor_pos_w_hands = anchor_pos_w[:, None, :].expand(-1, 2, -1)
    anchor_quat_w_hands = anchor_quat_w[:, None, :].expand(-1, 2, -1)
    hand_pos_b, hand_quat_b = subtract_frame_transforms(
      anchor_pos_w_hands,
      anchor_quat_w_hands,
      hand_pos_w,
      hand_quat_w,
    )
    hand_rot6d_b = _rot6d_from_matrix(matrix_from_quat(hand_quat_b))

    cmd_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
    cmd_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)

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
  def left_hand_body_index(self) -> int:
    return self._left_hand_idx

  @property
  def right_hand_body_index(self) -> int:
    return self._right_hand_idx

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

  def _update_metrics(self):
    super()._update_metrics()
    self._update_student_metrics()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    _debug_vis_hand_base_command(self, visualizer)


@dataclass(kw_only=True)
class DualViewMotionCommandCfg(MotionCommandCfg):
  """Configuration for combined student and teacher motion command views."""

  left_hand_body_name: str = ""
  right_hand_body_name: str = ""
  command_step_offsets: tuple[int, ...] = ()
  show_ghost: bool = True
  viz_scale: float = 0.5
  viz_z_offset: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if self.left_hand_body_name == "" or self.right_hand_body_name == "":
      raise ValueError(
        "`left_hand_body_name` and `right_hand_body_name` must be set for DualViewMotionCommandCfg."
      )
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for DualViewMotionCommandCfg."
      )
    return DualViewMotionCommand(self, env)
