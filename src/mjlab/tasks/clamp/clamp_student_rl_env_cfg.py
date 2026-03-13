"""CLAMP student-RL task configuration.

This module defines the task-level CLAMP student-RL configuration.
Robot-specific values are applied in config/<robot>/env_cfgs.py.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.clamp import mdp
from mjlab.tasks.clamp.clamp_teacher_env_cfg import (
  PUSH_VELOCITY_RANGE,
  make_clamp_teacher_env_cfg,
)
from mjlab.tasks.clamp.mdp import JointRefMotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def student_motion_command_kwargs() -> dict[str, object]:
  """Common kwargs shared by student joint-reference motion commands."""
  return {
    "entity_name": "robot",
    "resampling_time_range": (1.0e9, 1.0e9),
    "debug_vis": True,
    "pose_range": {
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    "velocity_range": PUSH_VELOCITY_RANGE,
    "joint_position_range": (-0.1, 0.1),
    # Set in robot cfg.
    "motion_file": "",
    "anchor_body_name": "",
    "body_names": (),
    "root_body_name": "",
    "sampling_mode": "adaptive",
  }


def make_clamp_student_rl_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CLAMP student-RL task configuration template."""
  cfg = make_clamp_teacher_env_cfg()

  command_terms = {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "motion"},
    ),
  }
  proprio_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  privileged_terms = {
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={"command_name": "motion"},
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={"command_name": "motion"},
    ),
    "feet_contact_mask": ObservationTermCfg(
      func=mdp.feet_contact_mask,
      params={"sensor_name": "feet_ground_contact"},
    ),
  }
  policy_terms = {**command_terms, **proprio_terms}
  critic_terms = {**policy_terms, **privileged_terms}
  cfg.observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.commands["motion"] = JointRefMotionCommandCfg(**student_motion_command_kwargs())

  return cfg
