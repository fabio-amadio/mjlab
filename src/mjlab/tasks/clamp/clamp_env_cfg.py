"""CLAMP Stage-A teacher task configuration.

This module defines the task-level CLAMP configuration (teacher stage).
Robot-specific values are applied in config/<robot>/env_cfgs.py.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.clamp import mdp
from mjlab.tasks.clamp.mdp import MotionCommandCfg
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}

TWIST_PUSH_VELOCITY_RANGE = {
  "x": (-1.0, 1.0),
  "y": (-1.0, 1.0),
  "z": (0.0, 0.0),
  "roll": (0.0, 0.0),
  "pitch": (0.0, 0.0),
  "yaw": (0.0, 0.0),
}

DEFAULT_TEACHER_FUTURE_STEPS = (
  1,
  5,
  10,
  15,
  20,
  25,
  30,
  35,
  40,
  45,
  50,
  55,
  60,
  65,
  70,
  75,
  80,
  85,
  90,
  95,
)

DEFAULT_TWIST_DOF_ERR_W = (
  1.0,
  0.8,
  0.8,
  1.0,
  0.5,
  0.5,
  1.0,
  0.8,
  0.8,
  1.0,
  0.5,
  0.5,
  0.6,
  0.6,
  0.6,
  0.8,
  0.8,
  0.8,
  1.0,
  0.4,
  0.4,
  0.4,
  0.8,
  0.8,
  0.8,
  1.0,
  0.4,
  0.4,
  0.4,
)


def make_clamp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CLAMP Stage-A teacher task configuration template."""

  ##
  # Observations (TWIST teacher structure)
  ##

  priv_mimic_terms = {
    "priv_motion_ref": ObservationTermCfg(
      func=mdp.motion_teacher_reference_obs,
      params={
        "command_name": "motion",
        "step_offsets": DEFAULT_TEACHER_FUTURE_STEPS,
        "key_body_names": (),  # Set in robot cfg.
      },
    ),
  }

  proprio_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  priv_info_terms = {
    "priv_info_root_pos": ObservationTermCfg(
      func=mdp.robot_anchor_pos,
      params={"command_name": "motion"},
    ),
    "priv_info_root_rpy": ObservationTermCfg(
      func=mdp.robot_anchor_rpy,
      params={"command_name": "motion"},
    ),
    "priv_info_key_body_pos": ObservationTermCfg(
      func=mdp.robot_key_body_pos_b,
      params={
        "command_name": "motion",
        "key_body_names": (),  # Set in robot cfg.
      },
    ),
    "priv_info_feet_contact_mask": ObservationTermCfg(
      func=mdp.feet_contact_mask,
      params={
        "sensor_name": "feet_ground_contact",  # Set/created in robot cfg.
      },
    ),
  }

  teacher_terms = {**priv_mimic_terms, **proprio_terms, **priv_info_terms}

  observations = {
    "policy": ObservationGroupCfg(
      terms=dict(teacher_terms),
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=dict(teacher_terms),
      concatenate_terms=True,
      enable_corruption=True,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0e9, 1.0e9),
      debug_vis=True,
      pose_range={
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.01, 0.01),
        "roll": (-0.1, 0.1),
        "pitch": (-0.1, 0.1),
        "yaw": (-0.2, 0.2),
      },
      velocity_range=VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      # Set in robot cfg.
      motion_file="",
      anchor_body_name="",
      body_names=(),
      sampling_mode="uniform",
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(4.0, 4.0),
      params={"velocity_range": TWIST_PUSH_VELOCITY_RANGE},
    ),
    "push_end_effector": EventTermCfg(
      func=mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(2.0, 2.0),
      params={
        "force_range": (-20.0, 20.0),
        "torque_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "field": "body_mass",
        "ranges": (-3.0, 3.0),
      },
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
        "operation": "add",
        "field": "body_ipos",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    "motor_strength": EventTermCfg(
      mode="startup",
      func=mdp.randomize_pd_gains,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "kp_range": (0.8, 1.2),
        "kd_range": (0.8, 1.2),
        "distribution": "uniform",
        "operation": "scale",
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=mdp.randomize_encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.01, 0.01),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set in robot cfg.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.1, 2.0),
        "shared_random": True,  # All foot geoms share the same friction.
      },
    ),
    "action_delay": EventTermCfg(
      mode="interval",
      func=mdp.sync_actuator_delays,
      interval_range_s=(0.02, 0.02),
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "lag_range": (0, 1),
      },
    ),
  }

  ##
  # Rewards (Stage-A teacher)
  ##

  rewards: dict[str, RewardTermCfg] = {
    "tracking_joint_dof": RewardTermCfg(
      func=mdp.motion_tracking_joint_dof,
      weight=0.6,
      params={
        "command_name": "motion",
        "pos_scale": 0.15,
        "dof_weights": DEFAULT_TWIST_DOF_ERR_W,
      },
    ),
    "tracking_joint_vel": RewardTermCfg(
      func=mdp.motion_tracking_joint_vel,
      weight=0.2,
      params={
        "command_name": "motion",
        "vel_scale": 0.01,
        "dof_weights": DEFAULT_TWIST_DOF_ERR_W,
      },
    ),
    "tracking_root_pose": RewardTermCfg(
      func=mdp.motion_tracking_root_pose,
      weight=0.6,
      params={"command_name": "motion", "root_pose_scale": 5.0, "in_world_frame": True},
    ),
    "tracking_root_vel": RewardTermCfg(
      func=mdp.motion_tracking_root_vel,
      weight=1.0,
      params={"command_name": "motion", "root_vel_scale": 1.0, "in_world_frame": False},
    ),
    "tracking_keybody_pos": RewardTermCfg(
      func=mdp.motion_tracking_keybody_pos,
      weight=2.0,
      params={
        "command_name": "motion",
        "key_body_names": (),  # Set in robot cfg.
        "key_body_pos_scale": 10.0,
        "in_world_frame": False,
      },
    ),
    "tracking_task_body_pos": RewardTermCfg(
      func=mdp.motion_tracking_task_body_pos,
      weight=0.0,
      params={
        "command_name": "motion",
        "task_body_names": (),  # Set in robot cfg.
        "task_body_pos_scale": 10.0,
      },
    ),
    "tracking_task_body_rot": RewardTermCfg(
      func=mdp.motion_tracking_task_body_rot,
      weight=0.0,
      params={
        "command_name": "motion",
        "task_body_names": (),  # Set in robot cfg.
        "task_body_rot_scale": 5.0,
      },
    ),
    "tracking_root_vel_xy": RewardTermCfg(
      func=mdp.motion_tracking_root_vel_xy,
      weight=0.0,
      params={"command_name": "motion", "in_world_frame": False},
    ),
    "tracking_root_ang_vel_yaw": RewardTermCfg(
      func=mdp.motion_tracking_root_ang_vel_yaw,
      weight=0.0,
      params={"command_name": "motion", "in_world_frame": False},
    ),
    "tracking_root_height": RewardTermCfg(
      func=mdp.motion_tracking_root_height,
      weight=0.0,
      params={"command_name": "motion", "height_scale": 5.0},
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "foot_body_names": (),  # Set in robot cfg.
        "command_name": "motion",
      },
    ),
    "feet_contact_forces": RewardTermCfg(
      func=mdp.feet_contact_forces,
      weight=-5.0e-4,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 100.0},
    ),
    "feet_stumble": RewardTermCfg(
      func=mdp.feet_stumble,
      weight=-1.25,
      params={"sensor_name": "feet_ground_contact", "ratio": 4.0},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_torque_limits": RewardTermCfg(
      func=mdp.dof_torque_limits,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot"), "soft_torque_limit": 0.95},
    ),
    "dof_vel": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "dof_acc": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-5.0e-8,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=5.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "target_air_time": 0.5,
      },
    ),
    "ang_vel_xy": RewardTermCfg(
      func=mdp.ang_vel_xy,
      weight=-0.01,
      params={"asset_cfg": SceneEntityCfg("robot")},
    ),
    "ankle_dof_acc": RewardTermCfg(
      func=mdp.joint_acc_l2,
      weight=-1.0e-7,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        ),
      },
    ),
    "ankle_dof_vel": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=-2.0e-4,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot",
          joint_names=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
        ),
      },
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "torso_contact": TerminationTermCfg(
      func=mdp.torso_contact_force,
      params={"sensor_name": "torso_ground_contact", "threshold": 1.0},
    ),
    "root_height_diff": TerminationTermCfg(
      func=mdp.bad_root_height_diff,
      params={"command_name": "motion", "threshold": 0.2},
    ),
    "roll_pitch": TerminationTermCfg(
      func=mdp.bad_roll_pitch,
      params={
        "roll_threshold": 1.0,
        "pitch_threshold": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "motion_end": TerminationTermCfg(
      func=mdp.motion_end,
      params={"command_name": "motion"},
    ),
    "root_lin_vel": TerminationTermCfg(
      func=mdp.root_lin_vel_too_large,
      params={"threshold": 5.0, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "pose_termination": TerminationTermCfg(
      func=mdp.pose_termination,
      params={
        "command_name": "motion",
        "threshold": 0.7,
        "body_names": (),  # Set in robot cfg.
        "in_world_frame": False,
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainImporterCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set in robot cfg.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=250,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
