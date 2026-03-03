"""CLAMP student-distillation task configuration.

This module defines the task-level CLAMP student-distillation configuration.
Robot-specific values are applied in config/<robot>/env_cfgs.py.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.clamp import mdp
from mjlab.tasks.clamp.clamp_student_rl_env_cfg import make_clamp_student_rl_env_cfg
from mjlab.tasks.clamp.clamp_teacher_env_cfg import (
  DEFAULT_TEACHER_FUTURE_STEPS,
  VELOCITY_RANGE,
)
from mjlab.tasks.clamp.mdp import TeacherStudentMotionCommandCfg


def make_clamp_student_distillation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CLAMP student-distillation task configuration template."""
  cfg = make_clamp_student_rl_env_cfg()

  policy_group = cfg.observations["policy"]
  policy_terms = dict(policy_group.terms)
  policy_terms["command"] = ObservationTermCfg(
    func=mdp.motion_student_command,
    params={"command_name": "motion"},
  )
  cfg.observations["policy"] = ObservationGroupCfg(
    terms=policy_terms,
    concatenate_terms=True,
    enable_corruption=policy_group.enable_corruption,
  )

  teacher_policy_terms = dict(policy_terms)
  teacher_policy_terms["command"] = ObservationTermCfg(
    func=mdp.motion_teacher_command,
    params={"command_name": "motion"},
  )
  cfg.observations["teacher_policy"] = ObservationGroupCfg(
    terms=teacher_policy_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.commands["motion"] = TeacherStudentMotionCommandCfg(
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
    motion_file="",
    anchor_body_name="",
    body_names=(),
    root_body_name="",
    left_hand_body_name="",
    right_hand_body_name="",
    command_step_offsets=DEFAULT_TEACHER_FUTURE_STEPS,
    sampling_mode="adaptive",
  )

  return cfg
