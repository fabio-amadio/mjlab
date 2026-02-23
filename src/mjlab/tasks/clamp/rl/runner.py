import os

import wandb
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.clamp.mdp import MotionCommand
from mjlab.tasks.clamp.mdp import MotionCommandCfg
from mjlab.tasks.clamp.rl.exporter import (
  attach_onnx_metadata,
  export_motion_policy_as_onnx,
)
from mjlab.tasks.clamp.rl.policy import ClampActorCriticMimic


class ClampOnPolicyRunner(MjlabOnPolicyRunner):
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
    # Register the CLAMP policy in RSL-RL runner scope so class_name eval resolves.
    rsl_on_policy_runner.ClampActorCriticMimic = ClampActorCriticMimic
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  @staticmethod
  def _infer_motion_obs_dims(env: VecEnv) -> tuple[int, int] | None:
    if not isinstance(env, RslRlVecEnvWrapper):
      return None
    env_unwrapped = env.unwrapped
    motion_cmd_cfg = env_unwrapped.cfg.commands.get("motion")
    if not isinstance(motion_cmd_cfg, MotionCommandCfg):
      return None
    command = env_unwrapped.command_manager.get_command("motion")
    if command is None:
      return None
    motion_obs_dim = int(command.shape[-1])
    motion_steps = 1
    if motion_cmd_cfg.command_mode in ("future_joint_ref", "future_joint_ref_anchor"):
      if len(motion_cmd_cfg.command_step_offsets) == 0:
        return None
      motion_steps = len(motion_cmd_cfg.command_step_offsets)
    return motion_obs_dim, motion_steps

  @classmethod
  def _configure_policy_cfg(cls, env: VecEnv, train_cfg: dict) -> None:
    policy_cfg = train_cfg.get("policy", {})
    policy_cfg.setdefault("class_name", "ClampActorCriticMimic")
    if policy_cfg.get("class_name") != "ClampActorCriticMimic":
      train_cfg["policy"] = policy_cfg
      return

    inferred_dims = cls._infer_motion_obs_dims(env)
    if inferred_dims is not None:
      motion_obs_dim, motion_steps = inferred_dims
      policy_cfg.setdefault("motion_obs_dim", motion_obs_dim)
      policy_cfg.setdefault("motion_steps", motion_steps)
    policy_cfg.setdefault("motion_latent_dim", 128)
    policy_cfg.setdefault("motion_channels", 20)
    policy_cfg.setdefault("layer_norm", True)
    train_cfg["policy"] = policy_cfg

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)

    motion_term = self.env.unwrapped.command_manager.get_term("motion")
    if isinstance(motion_term, MotionCommand) and motion_term.motion is None:
      # Multi-motion datasets use motion libraries; ONNX export is deferred.
      return

    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    if self.alg.policy.actor_obs_normalization:
      normalizer = self.alg.policy.actor_obs_normalizer
    else:
      normalizer = None
    export_motion_policy_as_onnx(
      self.env.unwrapped,
      self.alg.policy,
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    run_name = wandb.run.name if self.logger_type == "wandb" and wandb.run else "local"
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if self.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      # link the artifact registry to this run
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore
        self.registry_name = None
