import os

import numpy as np
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner
import wandb
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.clamp.mdp import MotionCommand, MotionCommandCfg
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
    """Infer flattened motion-command dim and number of future steps.

    Returns:
      `(motion_obs_dim, motion_steps)` for the `motion` command when available,
      otherwise `None`.
    """
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
    step_offsets = getattr(motion_cmd_cfg, "command_step_offsets", ())
    if len(step_offsets) > 0:
      motion_steps = len(step_offsets)
    return motion_obs_dim, motion_steps

  @staticmethod
  def _infer_obs_term_dims(
    env: VecEnv, target_term_name: str
  ) -> tuple[int, int] | None:
    """Infer `(flattened_dim, history_steps)` for one policy observation term.

    Notes:
      - `flattened_dim` is the final dim seen by the policy for that term
        (history already flattened if configured).
      - `history_steps` is the configured `ObservationTermCfg.history_length`.
    """
    if not isinstance(env, RslRlVecEnvWrapper):
      return None
    obs_manager = env.unwrapped.observation_manager
    group_terms = obs_manager.active_terms.get("policy", [])
    group_term_dims = obs_manager.group_obs_term_dim.get("policy", [])
    if len(group_terms) == 0 or len(group_terms) != len(group_term_dims):
      return None

    for term_name, term_dim in zip(group_terms, group_term_dims, strict=False):
      if term_name != target_term_name:
        continue
      term_cfg = obs_manager.get_term_cfg("policy", term_name)
      history_steps = max(int(term_cfg.history_length), 1)
      obs_dim = int(np.prod(term_dim))
      return obs_dim, history_steps
    return None

  @staticmethod
  def _validate_encoded_prefix_order(env: VecEnv) -> None:
    """Ensure policy-term order matches prefix slicing in `ClampActorCriticMimic`.

    The policy parser assumes encoded-prefix terms are ordered as:
    `command`, `short_io`, and optionally `long_io`.
    """
    if not isinstance(env, RslRlVecEnvWrapper):
      return
    obs_manager = env.unwrapped.observation_manager
    group_terms = obs_manager.active_terms.get("policy", [])
    expected_prefix = ["command", "short_io"]
    if "long_io" in group_terms:
      expected_prefix.append("long_io")
    actual_prefix = group_terms[: len(expected_prefix)]
    if actual_prefix != expected_prefix:
      raise ValueError(
        "Policy prefix encoding expects policy terms to start with "
        f"{expected_prefix}, got {actual_prefix}. "
        "Please reorder observation terms in CLAMP policy group."
      )

  @classmethod
  def _configure_policy_cfg(cls, env: VecEnv, train_cfg: dict) -> None:
    """Auto-fill CLAMP policy dimensions inferred from env observations/commands."""
    policy_cfg = train_cfg.get("policy", {})
    policy_cfg.setdefault("class_name", "ClampActorCriticMimic")
    if policy_cfg.get("class_name") != "ClampActorCriticMimic":
      train_cfg["policy"] = policy_cfg
      return
    cls._validate_encoded_prefix_order(env)

    inferred_dims = cls._infer_motion_obs_dims(env)
    if inferred_dims is not None:
      motion_obs_dim, motion_steps = inferred_dims
      policy_cfg.setdefault("motion_obs_dim", motion_obs_dim)
      policy_cfg.setdefault("motion_steps", motion_steps)
    short_io_dims = cls._infer_obs_term_dims(env, "short_io")
    if short_io_dims is not None:
      policy_cfg.setdefault("short_io_obs_dim", short_io_dims[0])

    long_io_dims = cls._infer_obs_term_dims(env, "long_io")
    if long_io_dims is not None:
      long_io_obs_dim, long_io_steps = long_io_dims
      policy_cfg.setdefault("long_io_obs_dim", long_io_obs_dim)
      policy_cfg.setdefault("long_io_steps", long_io_steps)
      policy_cfg.setdefault("long_io_latent_dim", 128)
    else:
      policy_cfg.setdefault("long_io_obs_dim", 0)
      policy_cfg.setdefault("long_io_steps", 1)
    policy_cfg.setdefault("motion_latent_dim", 128)
    policy_cfg.setdefault("print_model_structure", False)
    policy_cfg.setdefault("share_temporal_encoders", False)
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
