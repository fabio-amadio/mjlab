import rsl_rl.runners.distillation_runner as rsl_distillation_runner
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners import DistillationRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.clamp.mdp import MotionCommandCfg
from mjlab.tasks.clamp.rl.distillation_policy import ClampStudentTeacherDistill


class ClampDistillationRunner(DistillationRunner):
  """Distillation runner for CLAMP teacher->student setup."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ):
    self._configure_policy_cfg(env, train_cfg)
    # Register class in RSL-RL scope so DistillationRunner eval() can resolve it.
    rsl_distillation_runner.ClampStudentTeacherDistill = ClampStudentTeacherDistill
    super().__init__(env, train_cfg, log_dir, device)

  @staticmethod
  def _infer_teacher_motion_obs_dims(env: VecEnv) -> tuple[int, int] | None:
    if not isinstance(env, RslRlVecEnvWrapper):
      return None
    env_unwrapped = env.unwrapped
    motion_cmd_cfg = env_unwrapped.cfg.commands.get("motion")
    if not isinstance(motion_cmd_cfg, MotionCommandCfg):
      return None
    motion_term = env_unwrapped.command_manager.get_term("motion")
    teacher_command = getattr(motion_term, "teacher_command", None)
    if teacher_command is None:
      return None
    motion_obs_dim = int(teacher_command.shape[-1])
    motion_steps = len(getattr(motion_cmd_cfg, "command_step_offsets", ()))
    if motion_steps <= 0:
      motion_steps = 1
    return motion_obs_dim, motion_steps

  @classmethod
  def _configure_policy_cfg(cls, env: VecEnv, train_cfg: dict) -> None:
    policy_cfg = train_cfg.get("policy", {})
    policy_cfg.setdefault("class_name", "ClampStudentTeacherDistill")
    if policy_cfg.get("class_name") != "ClampStudentTeacherDistill":
      train_cfg["policy"] = policy_cfg
      return

    inferred_dims = cls._infer_teacher_motion_obs_dims(env)
    if inferred_dims is not None:
      teacher_motion_obs_dim, teacher_motion_steps = inferred_dims
      policy_cfg.setdefault("teacher_motion_obs_dim", teacher_motion_obs_dim)
      policy_cfg.setdefault("teacher_motion_steps", teacher_motion_steps)

    policy_cfg.setdefault("student_obs_normalization", True)
    policy_cfg.setdefault("teacher_obs_normalization", True)
    policy_cfg.setdefault("student_hidden_dims", (512, 512, 256, 128))
    policy_cfg.setdefault("activation", "elu")

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
