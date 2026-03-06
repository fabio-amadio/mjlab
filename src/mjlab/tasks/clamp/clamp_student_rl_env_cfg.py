"""CLAMP student-RL task configuration.

This module defines the task-level CLAMP student-RL configuration.
Robot-specific values are applied in config/<robot>/env_cfgs.py.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.clamp import mdp
from mjlab.tasks.clamp.clamp_teacher_env_cfg import (
  PUSH_VELOCITY_RANGE,
  make_clamp_teacher_env_cfg,
)
from mjlab.tasks.clamp.mdp import HandBaseMotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def student_motion_command_kwargs() -> dict[str, object]:
  """Common kwargs shared by student command configurations."""
  return {
    "entity_name": "robot",
    "resampling_time_range": (1.0e9, 1.0e9),
    "debug_vis": True,
    "show_ghost": True,
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
    "left_hand_body_name": "",
    "right_hand_body_name": "",
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

  cfg.commands["motion"] = HandBaseMotionCommandCfg(**student_motion_command_kwargs())

  # Keep full teacher reward set, then add student-focused terms for hands/base commands.
  student_tracking_rewards = {
    "hand_pos": RewardTermCfg(
      func=mdp.hand_position_tracking_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "hand_ori": RewardTermCfg(
      func=mdp.hand_orientation_tracking_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "base_lin_vel": RewardTermCfg(
      func=mdp.track_base_linear_velocity_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.5},
    ),
    "base_ang_vel": RewardTermCfg(
      func=mdp.track_base_angular_velocity_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.7},
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip_hand_base,
      weight=-0.05,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "command_threshold": 0.05,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing_hand_base,
      weight=-1e-5,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "command_threshold": 0.05,
      },
    ),
  }
  cfg.rewards = deepcopy(make_clamp_teacher_env_cfg().rewards)
  cfg.rewards.update(student_tracking_rewards)

  return cfg
