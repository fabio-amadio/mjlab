"""RL configuration for Unitree G1 CLAMP task."""

from mjlab.rl import (
  RslRlDistillationAlgorithmCfg,
  RslRlDistillationRunnerCfg,
  RslRlDistillationStudentTeacherCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_g1_clamp_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for CLAMP teacher training."""
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    policy=RslRlPpoActorCriticCfg(
      class_name="ClampActorCriticMimic",
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 512, 256, 128),
      critic_hidden_dims=(512, 512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_clamp_teacher",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_clamp_student_rl_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for CLAMP student-RL training."""
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    policy=RslRlPpoActorCriticCfg(
      class_name="ClampStudentActorCritic",
      init_noise_std=0.5,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 512, 256, 128),
      critic_hidden_dims=(512, 512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0025,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.005,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_clamp_student_rl",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_clamp_student_distillation_runner_cfg() -> RslRlDistillationRunnerCfg:
  """Create distillation runner configuration for CLAMP student distillation."""
  return RslRlDistillationRunnerCfg(
    seed=1,
    policy=RslRlDistillationStudentTeacherCfg(
      class_name="ClampStudentTeacherDistill",
      student_obs_normalization=True,
      teacher_obs_normalization=True,
      student_hidden_dims=(512, 512, 256, 128),
      activation="elu",
      init_noise_std=0.3,
    ),
    algorithm=RslRlDistillationAlgorithmCfg(
      class_name="Distillation",
      num_learning_epochs=5,
      gradient_length=15,
      learning_rate=5.0e-4,
      max_grad_norm=1.0,
      loss_type="mse",
      optimizer="adam",
    ),
    experiment_name="g1_clamp_student_distillation",
    save_interval=1000,
    num_steps_per_env=24,
    max_iterations=10_000,
    obs_groups={
      "policy": ("policy",),
      "teacher": ("teacher_policy",),
    },
  )
