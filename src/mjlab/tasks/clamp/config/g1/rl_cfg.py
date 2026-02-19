"""RL configuration for Unitree G1 CLAMP task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)

def unitree_g1_clamp_teacher_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for CLAMP Stage-A teacher training."""
  return RslRlOnPolicyRunnerCfg(
    seed=1,
    policy=RslRlPpoActorCriticCfg(
      class_name="ClampActorCriticMimic",
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 512, 256, 128),
      critic_hidden_dims=(512, 512, 256, 128),
      activation="swish",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=2.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.008,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_clamp_teacher_stage_a",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_002,
  )
