"""Unitree G1 CLAMP teacher environment configuration."""

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
from mjlab.tasks.clamp.clamp_student_distillation_env_cfg import (
  make_clamp_student_distillation_env_cfg,
)
from mjlab.tasks.clamp.clamp_student_rl_env_cfg import make_clamp_student_rl_env_cfg
from mjlab.tasks.clamp.clamp_teacher_env_cfg import make_clamp_teacher_env_cfg
from mjlab.tasks.clamp.mdp import (
  FutureJointRefAnchorMotionCommandCfg,
  HandBaseMotionCommandCfg,
  TeacherStudentMotionCommandCfg,
)

DEFAULT_CLAMP_MOTION_SOURCE = str(
  Path(__file__).resolve().with_name("motion_data_cfg.yaml")
)
G1_TRACKED_BODY_NAMES = (
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


def _apply_unitree_g1_overrides(
  cfg: ManagerBasedRlEnvCfg,
  play: bool,
) -> ManagerBasedRlEnvCfg:
  """Apply Unitree G1 robot/sensor/DR overrides to a CLAMP task template."""

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
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(
    motion_cmd,
    (
      FutureJointRefAnchorMotionCommandCfg,
      HandBaseMotionCommandCfg,
      TeacherStudentMotionCommandCfg,
    ),
  )
  motion_cmd.motion_file = DEFAULT_CLAMP_MOTION_SOURCE
  motion_cmd.anchor_body_name = "pelvis"
  motion_cmd.root_body_name = "pelvis"
  motion_cmd.body_names = G1_TRACKED_BODY_NAMES
  if isinstance(motion_cmd, (HandBaseMotionCommandCfg, TeacherStudentMotionCommandCfg)):
    motion_cmd.left_hand_body_name = "left_wrist_yaw_link"
    motion_cmd.right_hand_body_name = "right_wrist_yaw_link"
  motion_cmd.sampling_mode = "adaptive"

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_mass"].params["asset_cfg"].body_names = ("pelvis",)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["push_end_effector"].params["asset_cfg"].body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "pelvis"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.terminations.clear()
    cfg.events.pop("push_robot", None)
    cfg.events.pop("push_end_effector", None)
    cfg.events.pop("action_delay", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_flat_clamp_teacher_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 CLAMP teacher configuration."""
  return _apply_unitree_g1_overrides(make_clamp_teacher_env_cfg(), play=play)


def unitree_g1_flat_clamp_student_rl_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 CLAMP student-RL configuration."""
  return _apply_unitree_g1_overrides(make_clamp_student_rl_env_cfg(), play=play)


def unitree_g1_flat_clamp_student_distillation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 CLAMP student-distillation configuration."""
  return _apply_unitree_g1_overrides(
    make_clamp_student_distillation_env_cfg(), play=play
  )
