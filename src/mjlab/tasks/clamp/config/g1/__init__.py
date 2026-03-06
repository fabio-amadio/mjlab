from mjlab.tasks.clamp.rl import (
  ClampDistillPpoRunner,
  ClampDistillationRunner,
  ClampOnPolicyRunner,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_flat_clamp_student_distill_rl_env_cfg,
  unitree_g1_flat_clamp_student_distillation_env_cfg,
  unitree_g1_flat_clamp_student_rl_env_cfg,
  unitree_g1_flat_clamp_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_g1_clamp_student_distill_rl_runner_cfg,
  unitree_g1_clamp_student_distillation_runner_cfg,
  unitree_g1_clamp_student_rl_ppo_runner_cfg,
  unitree_g1_clamp_teacher_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-CLAMP-Teacher-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_clamp_teacher_env_cfg(),
  play_env_cfg=unitree_g1_flat_clamp_teacher_env_cfg(play=True),
  rl_cfg=unitree_g1_clamp_teacher_ppo_runner_cfg(),
  runner_cls=ClampOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-CLAMP-Student-RL-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_clamp_student_rl_env_cfg(),
  play_env_cfg=unitree_g1_flat_clamp_student_rl_env_cfg(play=True),
  rl_cfg=unitree_g1_clamp_student_rl_ppo_runner_cfg(),
  runner_cls=ClampOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-CLAMP-Student-Distillation-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_clamp_student_distillation_env_cfg(),
  play_env_cfg=unitree_g1_flat_clamp_student_distillation_env_cfg(play=True),
  rl_cfg=unitree_g1_clamp_student_distillation_runner_cfg(),
  runner_cls=ClampDistillationRunner,
)

register_mjlab_task(
  task_id="Mjlab-CLAMP-Student-Distill-RL-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_clamp_student_distill_rl_env_cfg(),
  play_env_cfg=unitree_g1_flat_clamp_student_distill_rl_env_cfg(play=True),
  rl_cfg=unitree_g1_clamp_student_distill_rl_runner_cfg(),
  runner_cls=ClampDistillPpoRunner,
)
