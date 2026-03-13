"""CLAMP student distill+RL task configuration.

This module defines the task-level CLAMP student distill+RL configuration.
Robot-specific values are applied in config/<robot>/env_cfgs.py.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.clamp import mdp
from mjlab.tasks.clamp.clamp_student_rl_env_cfg import (
  make_clamp_student_rl_env_cfg,
  student_motion_command_kwargs,
)
from mjlab.tasks.clamp.clamp_teacher_env_cfg import DEFAULT_TEACHER_FUTURE_STEPS
from mjlab.tasks.clamp.mdp import TeacherStudentMotionCommandCfg


def make_clamp_student_distill_rl_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CLAMP student distill+RL task configuration template."""
  cfg = make_clamp_student_rl_env_cfg()

  policy_group = cfg.observations["policy"]
  teacher_policy_terms = dict(policy_group.terms)
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
    **student_motion_command_kwargs(),
    future_sampling_step_offsets=DEFAULT_TEACHER_FUTURE_STEPS,
  )

  return cfg
