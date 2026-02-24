from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .motion_command import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_reference_exhausted(
  env: ManagerBasedRlEnv,
  command_name: str,
  include_future_horizon: bool = True,
) -> torch.Tensor:
  """Terminate when the current reference can no longer provide valid horizon."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  max_future_steps = 0
  if include_future_horizon:
    offsets = getattr(command.cfg, "command_step_offsets", ())
    if isinstance(offsets, tuple) and len(offsets) > 0:
      max_future_steps = max(int(offset) for offset in offsets)

  if command._uses_motion_library:
    assert command.motion_lib is not None
    horizon_s = float(max_future_steps) * float(env.step_dt)
    motion_lengths = command.motion_lib.get_motion_length(command.motion_ids)
    return command._current_times_s() + horizon_s >= motion_lengths

  assert command.motion is not None
  return command.time_steps + max_future_steps >= command.motion.time_step_total
