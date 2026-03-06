import copy
import os
from typing import cast

import torch.nn as nn
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.clamp.mdp import MotionCommand
from mjlab.tasks.clamp.rl.distillation_policy import ClampStudentTeacherDistill
from mjlab.tasks.clamp.rl.policy import ClampActorCriticMimic, MotionEncoder
from mjlab.tasks.clamp.rl.student_policy import ClampStudentActorCritic


def export_clamp_policy_as_onnx(
  env: ManagerBasedRlEnv,
  policy: object,
  path: str,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_obs_dim = _infer_policy_obs_dim(env)
  policy_exporter = _OnnxClampInferenceExporter(policy, verbose=verbose)
  policy_exporter.export(path, filename, policy_obs_dim)


def _flatten_obs_dim(dim: int | tuple[int, ...]) -> int:
  if isinstance(dim, int):
    return int(dim)
  out = 1
  for d in dim:
    out *= int(d)
  return out


def _infer_policy_obs_dim(env: ManagerBasedRlEnv) -> int:
  group_dim = env.observation_manager.group_obs_dim["policy"]
  if isinstance(group_dim, list):
    return int(sum(_flatten_obs_dim(dim) for dim in group_dim))
  return int(_flatten_obs_dim(group_dim))


class _OnnxClampInferenceExporter(nn.Module):
  def __init__(self, policy: object, verbose: bool = False):
    super().__init__()
    self.verbose = verbose
    self.policy_type: str

    if isinstance(policy, ClampActorCriticMimic):
      self.policy_type = "teacher"
      self.actor = copy.deepcopy(policy.actor)
      self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
      self.motion_encoder: MotionEncoder | None = copy.deepcopy(
        policy.actor_motion_encoder
      )
      self.motion_obs_dim = int(policy.motion_obs_dim)
      self.single_motion_obs_dim = int(policy.single_motion_obs_dim)
    elif isinstance(policy, ClampStudentActorCritic):
      self.policy_type = "student_rl"
      self.actor = copy.deepcopy(policy.actor)
      self.normalizer = copy.deepcopy(policy.actor_obs_normalizer)
      self.motion_encoder = None
      self.motion_obs_dim = 0
      self.single_motion_obs_dim = 0
    elif isinstance(policy, ClampStudentTeacherDistill):
      self.policy_type = "student_distillation"
      self.actor = copy.deepcopy(policy.student)
      self.normalizer = copy.deepcopy(policy.student_obs_normalizer)
      self.motion_encoder = None
      self.motion_obs_dim = 0
      self.single_motion_obs_dim = 0
    else:
      raise TypeError(
        "Unsupported policy type for CLAMP ONNX export: "
        f"{type(policy).__name__}."
      )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    actor_obs = self.normalizer(obs)
    if self.policy_type == "teacher":
      assert self.motion_encoder is not None
      motion_obs = actor_obs[:, : self.motion_obs_dim]
      remainder = actor_obs[:, self.motion_obs_dim :]
      motion_latent = self.motion_encoder(motion_obs)
      first_step_motion = motion_obs[:, : self.single_motion_obs_dim]
      actor_input = torch.cat((remainder, first_step_motion, motion_latent), dim=-1)
      return self.actor(actor_input)
    return self.actor(actor_obs)

  def export(self, path: str, filename: str, policy_obs_dim: int) -> None:
    self.to("cpu")
    self.eval()
    obs = torch.zeros(1, policy_obs_dim)
    torch.onnx.export(
      self,
      obs,
      os.path.join(path, filename),
      export_params=True,
      opset_version=11,
      verbose=self.verbose,
      input_names=["obs"],
      output_names=["actions"],
      dynamic_axes={},
      dynamo=False,
    )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach CLAMP-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)

  # Get base metadata common to all tasks.
  metadata = get_base_metadata(env, run_path)

  # Add CLAMP-specific metadata.
  motion_term = env.command_manager.get_term("motion")
  assert isinstance(motion_term, MotionCommand)
  motion_term_cfg = motion_term.cfg
  metadata.update(
    {
      "anchor_body_name": motion_term_cfg.anchor_body_name,
      "body_names": list(motion_term_cfg.body_names),
    }
  )

  attach_metadata_to_onnx(onnx_path, metadata)
