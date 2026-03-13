"""CLAMP student-distillation task configuration.

This module defines the task-level CLAMP student-distillation configuration.
Robot-specific values are applied in config/<robot>/env_cfgs.py.
"""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.clamp import mdp
from mjlab.tasks.clamp.clamp_student_rl_env_cfg import (
  make_clamp_student_rl_env_cfg,
  student_motion_command_kwargs,
)
from mjlab.tasks.clamp.clamp_teacher_env_cfg import (
  DEFAULT_TEACHER_FUTURE_STEPS,
  make_clamp_teacher_env_cfg,
)
from mjlab.tasks.clamp.mdp import TeacherStudentMotionCommandCfg


def make_clamp_student_distillation_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create CLAMP student-distillation task configuration template."""
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
  # Keep reward naming/weights identical to teacher for direct W&B comparison.
  cfg.rewards = deepcopy(make_clamp_teacher_env_cfg().rewards)

  return cfg
