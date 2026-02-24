"""Unitree G1 CLAMP Stage-A environment configuration."""

from copy import deepcopy
from pathlib import Path

from mjlab.actuator import DelayedActuatorCfg
from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.clamp.clamp_env_cfg import make_clamp_env_cfg
from mjlab.tasks.clamp.mdp import FutureJointRefAnchorMotionCommandCfg

DEFAULT_CLAMP_STAGE_A_MOTION_SOURCE = str(
  Path(__file__).resolve().with_name("motion_data_cfg.yaml")
)


def unitree_g1_flat_clamp_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 CLAMP Stage-A teacher configuration."""
  cfg = make_clamp_env_cfg()

  robot_cfg = get_g1_robot_cfg()
  assert robot_cfg.articulation is not None
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=tuple(
      DelayedActuatorCfg(
        base_cfg=deepcopy(actuator_cfg),
        delay_target="position",
        delay_min_lag=0,
        delay_max_lag=1,
        # Keep lag fixed unless sync_actuator_delays resamples it.
        delay_hold_prob=1.0,
      )
      for actuator_cfg in robot_cfg.articulation.actuators
    ),
    soft_joint_pos_limit_factor=robot_cfg.articulation.soft_joint_pos_limit_factor,
  )
  cfg.scene.entities = {"robot": robot_cfg}

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  torso_ground_cfg = ContactSensorCfg(
    name="torso_ground_contact",
    primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, torso_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, FutureJointRefAnchorMotionCommandCfg)
  motion_cmd.motion_file = DEFAULT_CLAMP_STAGE_A_MOTION_SOURCE
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )
  motion_cmd.sampling_mode = "uniform"

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_mass"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["push_end_effector"].params["asset_cfg"].body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.terminations["pose_termination"].params["body_names"] = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_knee_link",
    "right_knee_link",
    "left_elbow_link",
    "right_elbow_link",
  )

  cfg.viewer.body_name = "pelvis"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_end_effector", None)
    cfg.events.pop("action_delay", None)
    cfg.terminations.pop("motion_end", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
