"""CLAMP teacher task configuration.

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
from mjlab.tasks.clamp.mdp import FutureJointRefAnchorMotionCommandCfg
from mjlab.tasks.velocity import mdp as velocity_mdp
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

PUSH_VELOCITY_RANGE = {
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


def make_clamp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CLAMP teacher task configuration template."""

  ##
  # Observations
  ##

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
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.1, n_max=0.1),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.1, n_max=0.1),
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

  observations = {
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
    "motion": FutureJointRefAnchorMotionCommandCfg(
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
      command_step_offsets=DEFAULT_TEACHER_FUTURE_STEPS,
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
      params={"velocity_range": PUSH_VELOCITY_RANGE},
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
      },
    ),
    "tracking_joint_vel": RewardTermCfg(
      func=mdp.motion_tracking_joint_vel,
      weight=0.2,
      params={
        "command_name": "motion",
        "vel_scale": 0.01,
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
    "feet_slip": RewardTermCfg(
      func=velocity_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        # Motion command is not a velocity command; keep this always active.
        "command_threshold": -1.0,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set in robot cfg.
      },
    ),
    "soft_landing": RewardTermCfg(
      func=velocity_mdp.soft_landing,
      weight=-1.0e-5,
      params={
        "sensor_name": "feet_ground_contact",
        # Motion command is not a velocity command; keep this always active.
        "command_name": "motion",
        "command_threshold": -1.0,
      },
    ),
    "self_collisions": RewardTermCfg(
      func=velocity_mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision"},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_torques_l2": RewardTermCfg(
      func=mdp.joint_torques_l2,
      # Calibrated for CLAMP scale where joint_torques_l2 is O(1e3).
      weight=-1.0e-4,
      params={"asset_cfg": SceneEntityCfg("robot")},
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
    "feet_air_time_twist": RewardTermCfg(
      func=mdp.feet_air_time_twist,
      weight=5.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "motion",
        "target_air_time": 0.5,
      },
    ),
    "ang_vel_xy": RewardTermCfg(
      func=velocity_mdp.body_angular_velocity_penalty,
      weight=-0.01,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # Set in robot cfg.
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "illegal_contact": TerminationTermCfg(
      func=velocity_mdp.illegal_contact,
      params={"sensor_name": "torso_ground_contact"},
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
