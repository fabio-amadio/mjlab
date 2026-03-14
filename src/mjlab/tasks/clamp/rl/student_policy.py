from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import ActorCritic


class ClampStudentActorCritic(ActorCritic):
  """ActorCritic with actor-only initialization from CLAMP distillation checkpoints."""

  _DISTILL_STUDENT_PREFIX = "student."
  _DISTILL_NORMALIZER_PREFIX = "student_obs_normalizer."

  def load_state_dict(
    self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
  ) -> bool:
    # Standard PPO checkpoint path.
    if not any(key.startswith(self._DISTILL_STUDENT_PREFIX) for key in state_dict):
      return super().load_state_dict(state_dict, strict=strict)

    actor_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
      if key.startswith(self._DISTILL_STUDENT_PREFIX):
        mapped_key = f"actor.{key[len(self._DISTILL_STUDENT_PREFIX) :]}"
        actor_state_dict[mapped_key] = value
      elif key.startswith(self._DISTILL_NORMALIZER_PREFIX):
        mapped_key = (
          f"actor_obs_normalizer.{key[len(self._DISTILL_NORMALIZER_PREFIX) :]}"
        )
        actor_state_dict[mapped_key] = value

    if len(actor_state_dict) == 0:
      raise ValueError(
        "Distillation checkpoint does not contain student actor parameters."
      )

    model_keys = set(self.state_dict().keys())
    required_actor_keys = {
      key
      for key in model_keys
      if key.startswith("actor.") or key.startswith("actor_obs_normalizer.")
    }

    provided_actor_keys = set(actor_state_dict.keys())
    missing_actor_keys = sorted(required_actor_keys - provided_actor_keys)
    if len(missing_actor_keys) > 0:
      raise ValueError(
        "Distillation checkpoint is missing required actor-side keys: "
        f"{missing_actor_keys[:20]}"
      )

    unexpected_actor_keys = sorted(provided_actor_keys - model_keys)
    if len(unexpected_actor_keys) > 0:
      raise ValueError(
        "Distillation checkpoint contains unexpected keys for student actor: "
        f"{unexpected_actor_keys[:20]}"
      )

    incompatible = nn.Module.load_state_dict(self, actor_state_dict, strict=False)
    unresolved_actor_keys = [
      key for key in incompatible.missing_keys if key in required_actor_keys
    ]
    if len(unresolved_actor_keys) > 0:
      raise ValueError(
        "Failed to load all required actor-side parameters. Missing after load: "
        f"{unresolved_actor_keys[:20]}"
      )

    # Actor initialized from distillation student, critic intentionally random.
    return False
