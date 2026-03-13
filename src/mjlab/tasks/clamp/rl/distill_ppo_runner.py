from typing import cast

import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
from mjlab.tasks.clamp.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.clamp.rl.distill_ppo_algorithm import DistillPPO
from mjlab.tasks.clamp.rl.distill_ppo_policy import ClampStudentDistillPpoActorCritic
from mjlab.tasks.clamp.rl.policy import ClampActorCriticMimic
from mjlab.tasks.clamp.rl.runner import ClampOnPolicyRunner
from mjlab.tasks.clamp.rl.student_policy import ClampStudentActorCritic


class ClampDistillPpoRunner(ClampOnPolicyRunner):
  """CLAMP on-policy runner for mixed distillation + PPO training."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    self._configure_policy_cfg(env, train_cfg)
    rsl_on_policy_runner.ClampActorCriticMimic = ClampActorCriticMimic
    rsl_on_policy_runner.ClampStudentActorCritic = ClampStudentActorCritic
    rsl_on_policy_runner.ClampStudentDistillPpoActorCritic = (
      ClampStudentDistillPpoActorCritic
    )
    rsl_on_policy_runner.DistillPPO = DistillPPO
    MjlabOnPolicyRunner.__init__(self, env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  @staticmethod
  def _infer_teacher_motion_obs_dims(env: VecEnv) -> tuple[int, int] | None:
    if not isinstance(env, RslRlVecEnvWrapper):
      return None
    env_unwrapped = env.unwrapped
    motion_cmd_cfg = env_unwrapped.cfg.commands.get("motion")
    if not isinstance(motion_cmd_cfg, MotionCommandCfg):
      return None
    motion_term = cast(MotionCommand, env_unwrapped.command_manager.get_term("motion"))
    if motion_term is None or not motion_term.has_command_representation("teacher"):
      return None
    teacher_command = motion_term.get_command_representation("teacher")
    motion_obs_dim = int(teacher_command.shape[-1])
    motion_steps = len(motion_term.future_sampling_step_offsets)
    if motion_steps <= 0:
      motion_steps = 1
    return motion_obs_dim, motion_steps

  @classmethod
  def _configure_policy_cfg(cls, env: VecEnv, train_cfg: dict) -> None:
    policy_cfg = train_cfg.get("policy", {})
    policy_cfg.setdefault("class_name", "ClampStudentDistillPpoActorCritic")
    if policy_cfg.get("class_name") != "ClampStudentDistillPpoActorCritic":
      return super()._configure_policy_cfg(env, train_cfg)

    inferred_dims = cls._infer_teacher_motion_obs_dims(env)
    if inferred_dims is not None:
      teacher_motion_obs_dim, teacher_motion_steps = inferred_dims
      policy_cfg.setdefault("teacher_motion_obs_dim", teacher_motion_obs_dim)
      policy_cfg.setdefault("teacher_motion_steps", teacher_motion_steps)

    policy_cfg.setdefault("teacher_obs_normalization", True)
    policy_cfg.setdefault("teacher_actor_hidden_dims", (512, 512, 256, 128))
    policy_cfg.setdefault("teacher_critic_hidden_dims", (512, 512, 256, 128))
    policy_cfg.setdefault("teacher_activation", "elu")
    policy_cfg.setdefault("teacher_motion_latent_dim", 128)
    policy_cfg.setdefault("teacher_motion_proj_channels", 60)
    policy_cfg.setdefault("teacher_motion_conv_channels", (40, 20))
    policy_cfg.setdefault("teacher_motion_conv_kernel_sizes", (6, 4))
    policy_cfg.setdefault("teacher_motion_conv_strides", (2, 2))
    policy_cfg.setdefault("teacher_layer_norm", True)
    policy_cfg.setdefault("teacher_print_model_structure", False)
    train_cfg["policy"] = policy_cfg
