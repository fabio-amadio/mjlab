from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from .motion_command import MotionCommand


def debug_visualize_motion_command(
  command: MotionCommand,
  visualizer: DebugVisualizer,
  desired_frame_colors: tuple[tuple[float, float, float], ...],
) -> None:
  env_indices = visualizer.get_env_indices(command.num_envs)
  if not env_indices:
    return

  if command.cfg.viz.mode == "ghost":
    if command._ghost_model is None:
      command._ghost_model = copy.deepcopy(command._env.sim.mj_model)
      command._ghost_model.geom_rgba[:] = command._ghost_color

    entity = command._env.scene[command.cfg.entity_name]
    indexing = entity.indexing
    free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = indexing.joint_q_adr.cpu().numpy()

    for batch in env_indices:
      qpos = np.zeros(command._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = command.anchor_pos_w[batch].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = command.anchor_quat_w[batch].cpu().numpy()
      qpos[joint_q_adr] = command.joint_pos[batch].cpu().numpy()
      visualizer.add_ghost_mesh(qpos, model=command._ghost_model, label=f"ghost_{batch}")

  elif command.cfg.viz.mode == "frames":
    for batch in env_indices:
      desired_body_pos = command.body_pos_w[batch].cpu().numpy()
      desired_body_quat = command.body_quat_w[batch]
      desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

      current_body_pos = command.robot_body_pos_w[batch].cpu().numpy()
      current_body_quat = command.robot_body_quat_w[batch]
      current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

      for i, body_name in enumerate(command.cfg.body_names):
        visualizer.add_frame(
          position=desired_body_pos[i],
          rotation_matrix=desired_body_rotm[i],
          scale=0.08,
          label=f"desired_{body_name}_{batch}",
          axis_colors=desired_frame_colors,
        )
        visualizer.add_frame(
          position=current_body_pos[i],
          rotation_matrix=current_body_rotm[i],
          scale=0.12,
          label=f"current_{body_name}_{batch}",
        )

      desired_anchor_pos = command.anchor_pos_w[batch].cpu().numpy()
      desired_anchor_quat = command.anchor_quat_w[batch]
      desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=desired_anchor_pos,
        rotation_matrix=desired_rotation_matrix,
        scale=0.1,
        label=f"desired_anchor_{batch}",
        axis_colors=desired_frame_colors,
      )

      current_anchor_pos = command.robot_anchor_pos_w[batch].cpu().numpy()
      current_anchor_quat = command.robot_anchor_quat_w[batch]
      current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=current_anchor_pos,
        rotation_matrix=current_rotation_matrix,
        scale=0.15,
        label=f"current_anchor_{batch}",
      )
