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
from mjlab.tasks.tracking import mdp as tracking_mdp
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


def make_clamp_teacher_env_cfg() -> ManagerBasedRlEnvCfg:
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
      root_body_name="",
      command_step_offsets=DEFAULT_TEACHER_FUTURE_STEPS,
      sampling_mode="adaptive",
    )
  }

  ##
  # Events
  ##

  events: dict[str, EventTermCfg] = {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": VELOCITY_RANGE},
    ),
    "push_end_effector": EventTermCfg(
      func=mdp.apply_external_force_torque,
      mode="interval",
      interval_range_s=(2.0, 2.0),
      params={
        "force_range": (-10.0, 10.0),
        "torque_range": (-1.0, 1.0),
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
          0: (-0.025, 0.025),
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
        "ranges": (0.3, 1.2),
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
  # Rewards (CLAMP teacher)
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_global_root_pos": RewardTermCfg(
      func=tracking_mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=tracking_mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=tracking_mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "self_collisions": RewardTermCfg(
      func=tracking_mdp.self_collision_cost,
      weight=-10.0,
      params={"sensor_name": "self_collision"},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    "anchor_pos": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=tracking_mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),  # Set per-robot.
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
