from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .motion_command import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _rot6d_from_matrix(rotm: torch.Tensor) -> torch.Tensor:
  """Encode rotation matrix as first two columns: [c1(3), c2(3)]."""
  first_two_cols = rotm[..., :, :2]
  return first_two_cols.transpose(-2, -1).reshape(*rotm.shape[:-2], 6)


def _matrix_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
  """Decode 6D orientation represented as [c1(3), c2(3)] into rotation matrix."""
  c1_raw = rot6d[..., 0:3]
  c2_raw = rot6d[..., 3:6]
  c1 = torch.nn.functional.normalize(c1_raw, dim=-1)
  proj = torch.sum(c1 * c2_raw, dim=-1, keepdim=True)
  c2 = torch.nn.functional.normalize(c2_raw - proj * c1, dim=-1)
  c3 = torch.cross(c1, c2, dim=-1)
  return torch.stack((c1, c2, c3), dim=-1)


def _debug_vis_hand_base_command(command, visualizer: DebugVisualizer) -> None:
  env_indices = visualizer.get_env_indices(command.num_envs)
  if not env_indices:
    return

  show_ghost = getattr(command.cfg, "show_ghost", True)
  free_joint_q_adr = None
  joint_q_adr = None
  if show_ghost:
    if command._ghost_model is None:
      command._ghost_model = copy.deepcopy(command._env.sim.mj_model)
      command._ghost_model.geom_rgba[:] = command._ghost_color

    entity = command._env.scene[command.cfg.entity_name]
    indexing = entity.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()

  left_idx = command.left_hand_body_index
  right_idx = command.right_hand_body_index
  left_robot_idx = int(command.robot_body_indexes[left_idx].item())
  right_robot_idx = int(command.robot_body_indexes[right_idx].item())

  current_hand_pos_w = torch.stack(
    [
      command.robot.data.body_link_pos_w[:, left_robot_idx],
      command.robot.data.body_link_pos_w[:, right_robot_idx],
    ],
    dim=1,
  )
  current_hand_quat_w = torch.stack(
    [
      command.robot.data.body_link_quat_w[:, left_robot_idx],
      command.robot.data.body_link_quat_w[:, right_robot_idx],
    ],
    dim=1,
  )
  current_hand_rotm_w = matrix_from_quat(current_hand_quat_w)

  student_cmd = command.command
  desired_hand_pos_b = student_cmd[:, :6].reshape(command.num_envs, 2, 3)
  desired_hand_rot6d_b = student_cmd[:, 6:18].reshape(command.num_envs, 2, 6)
  desired_hand_rotm_b = _matrix_from_rot6d(
    desired_hand_rot6d_b.reshape(command.num_envs * 2, 6)
  ).reshape(command.num_envs, 2, 3, 3)

  anchor_pos_w = command.robot_anchor_pos_w
  anchor_quat_w = command.robot_anchor_quat_w
  anchor_rotm_w = matrix_from_quat(anchor_quat_w)
  anchor_quat_w_hands = anchor_quat_w[:, None, :].expand(-1, 2, -1)
  desired_hand_pos_w = anchor_pos_w[:, None, :] + quat_apply(
    anchor_quat_w_hands, desired_hand_pos_b
  )
  desired_hand_rotm_w = anchor_rotm_w[:, None, :, :] @ desired_hand_rotm_b

  actual_lin_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot_anchor_lin_vel_w
  )
  actual_ang_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot_anchor_ang_vel_w
  )

  hand_names = ("left_hand", "right_hand")
  arrow_scale = float(getattr(command.cfg, "viz_scale", 0.5))
  z_offset = float(getattr(command.cfg, "viz_z_offset", 0.2))

  for batch in env_indices:
    if show_ghost:
      assert free_joint_q_adr is not None and joint_q_adr is not None
      qpos = np.zeros(command._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = command.root_pos_w[batch].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = command.root_quat_w[batch].cpu().numpy()
      qpos[joint_q_adr] = command.joint_pos[batch].cpu().numpy()
      visualizer.add_ghost_mesh(
        qpos, model=command._ghost_model, label=f"student_ghost_{batch}"
      )

    for hand_i, hand_name in enumerate(hand_names):
      visualizer.add_frame(
        position=desired_hand_pos_w[batch, hand_i].cpu().numpy(),
        rotation_matrix=desired_hand_rotm_w[batch, hand_i].cpu().numpy(),
        scale=0.12,
        label=f"desired_{hand_name}_{batch}",
        axis_colors=_DESIRED_FRAME_COLORS,
      )
      visualizer.add_frame(
        position=current_hand_pos_w[batch, hand_i].cpu().numpy(),
        rotation_matrix=current_hand_rotm_w[batch, hand_i].cpu().numpy(),
        scale=0.14,
        label=f"current_{hand_name}_{batch}",
      )

    base_pos_w = anchor_pos_w[batch].cpu().numpy()
    base_rotm_w = anchor_rotm_w[batch].cpu().numpy()
    if np.linalg.norm(base_pos_w) < 1e-6:
      continue

    cmd_vx = float(student_cmd[batch, 18].item())
    cmd_vy = float(student_cmd[batch, 19].item())
    cmd_wz = float(student_cmd[batch, 20].item())
    act_vx = float(actual_lin_vel_b[batch, 0].item())
    act_vy = float(actual_lin_vel_b[batch, 1].item())
    act_wz = float(actual_ang_vel_b[batch, 2].item())

    def local_to_world(
      local_vec: np.ndarray,
      base_pos_w: np.ndarray = base_pos_w,
      base_rotm_w: np.ndarray = base_rotm_w,
    ) -> np.ndarray:
      return base_pos_w + base_rotm_w @ local_vec

    cmd_lin_from = local_to_world(np.array([0.0, 0.0, z_offset]) * arrow_scale)
    cmd_lin_to = local_to_world(
      (np.array([0.0, 0.0, z_offset]) + np.array([cmd_vx, cmd_vy, 0.0])) * arrow_scale
    )
    visualizer.add_arrow(
      cmd_lin_from,
      cmd_lin_to,
      color=(0.2, 0.2, 0.6, 0.6),
      width=0.015,
      label=f"cmd_lin_{batch}",
    )

    cmd_ang_from = cmd_lin_from
    cmd_ang_to = local_to_world(
      (np.array([0.0, 0.0, z_offset]) + np.array([0.0, 0.0, cmd_wz])) * arrow_scale
    )
    visualizer.add_arrow(
      cmd_ang_from,
      cmd_ang_to,
      color=(0.2, 0.6, 0.2, 0.6),
      width=0.015,
      label=f"cmd_ang_{batch}",
    )

    act_lin_from = cmd_lin_from
    act_lin_to = local_to_world(
      (np.array([0.0, 0.0, z_offset]) + np.array([act_vx, act_vy, 0.0])) * arrow_scale
    )
    visualizer.add_arrow(
      act_lin_from,
      act_lin_to,
      color=(0.0, 0.6, 1.0, 0.7),
      width=0.015,
      label=f"act_lin_{batch}",
    )

    act_ang_from = cmd_lin_from
    act_ang_to = local_to_world(
      (np.array([0.0, 0.0, z_offset]) + np.array([0.0, 0.0, act_wz])) * arrow_scale
    )
    visualizer.add_arrow(
      act_ang_from,
      act_ang_to,
      color=(0.0, 1.0, 0.4, 0.7),
      width=0.015,
      label=f"act_ang_{batch}",
    )


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
    student_cmd = self.command
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

  def _build_command(self) -> torch.Tensor:
    # Current desired reference frame (single step).
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

    # Desired anchor velocity from reference clip, expressed in reference anchor frame.
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

  def _refresh_command_cache(self) -> None:
    self._command_cache = self._build_command()

  @property
  def left_hand_body_index(self) -> int:
    return self._left_hand_idx

  @property
  def right_hand_body_index(self) -> int:
    return self._right_hand_idx

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

  def _update_metrics(self):
    super()._update_metrics()
    self._update_student_metrics()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    _debug_vis_hand_base_command(self, visualizer)


@dataclass(kw_only=True)
class HandBaseMotionCommandCfg(MotionCommandCfg):
  """Configuration for student hand+base command representation."""

  left_hand_body_name: str = ""
  right_hand_body_name: str = ""
  show_ghost: bool = True
  viz_scale: float = 0.5
  viz_z_offset: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if self.left_hand_body_name == "" or self.right_hand_body_name == "":
      raise ValueError(
        "`left_hand_body_name` and `right_hand_body_name` must be set for HandBaseMotionCommandCfg."
      )
    return HandBaseMotionCommand(self, env)
