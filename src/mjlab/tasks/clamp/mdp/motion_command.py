from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_error_magnitude,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .motion_debug import debug_visualize_motion_command
from .motion_indexing import build_name_to_index, resolve_motion_body_names
from .motion_library import MotionFrameBatch, MotionLoader, NpzMotionLibrary
from .motion_sampling import (
  adaptive_sampling,
  resample_command,
  uniform_sampling,
  update_command,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._uses_motion_library = False

    self.motion: MotionLoader | None = None
    self.motion_lib: NpzMotionLibrary | None = None
    self._current_motion_frame: MotionFrameBatch | None = None

    source = Path(self.cfg.motion_file)
    if source.suffix == ".npz":
      if not source.is_file():
        raise ValueError(f"Motion npz file does not exist: {self.cfg.motion_file}")
      self.motion = MotionLoader(self.cfg.motion_file, device=self.device)
    else:
      self.motion_lib = NpzMotionLibrary(
        self.cfg.motion_file,
        device=self.device,
        show_progress=self.cfg.show_motion_load_progress,
      )
      self._uses_motion_library = True

    motion_body_names = resolve_motion_body_names(self)
    motion_name_to_index = build_name_to_index(motion_body_names, source="motion")
    robot_body_names = tuple(self.robot.body_names)
    robot_name_to_index = build_name_to_index(robot_body_names, source="robot")

    required_body_names = list(
      dict.fromkeys((self.cfg.anchor_body_name, *cfg.body_names))
    )
    missing_motion = [n for n in required_body_names if n not in motion_name_to_index]
    missing_robot = [n for n in required_body_names if n not in robot_name_to_index]
    if missing_motion or missing_robot:
      error_lines = ["Body name mapping failed while initializing CLAMP MotionCommand."]
      if missing_motion:
        error_lines.append(f"  Missing in motion reference: {missing_motion}")
      if missing_robot:
        error_lines.append(f"  Missing in robot model: {missing_robot}")
      raise ValueError("\n".join(error_lines))

    self.robot_anchor_body_index = robot_name_to_index[self.cfg.anchor_body_name]
    self.motion_anchor_body_index = motion_name_to_index[self.cfg.anchor_body_name]
    self.robot_body_indexes = torch.tensor(
      [robot_name_to_index[name] for name in self.cfg.body_names],
      dtype=torch.long,
      device=self.device,
    )
    self.motion_body_indexes = torch.tensor(
      [motion_name_to_index[name] for name in self.cfg.body_names],
      dtype=torch.long,
      device=self.device,
    )
    # Backward-compatible alias used in downstream utilities.
    self.body_indexes = self.robot_body_indexes
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_time_offsets = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    if self._uses_motion_library:
      assert self.motion_lib is not None
      max_motion_len_s = float(torch.max(self.motion_lib.motion_lengths_s).item())
      self.bin_count = max(int(max_motion_len_s / max(env.step_dt, 1e-6)) + 1, 1)
      self._refresh_motion_frame()
    else:
      assert self.motion is not None
      self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1

    self.bin_failed_count: torch.Tensor | None = None
    self._current_bin_failed: torch.Tensor | None = None
    self.motion_failed_count: torch.Tensor | None = None
    self._current_motion_failed: torch.Tensor | None = None
    self.phase_failed_count: torch.Tensor | None = None
    self._current_phase_failed: torch.Tensor | None = None
    if self._uses_motion_library:
      assert self.motion_lib is not None
      num_motions = self.motion_lib.num_motions()
      self.motion_failed_count = torch.zeros(
        num_motions, dtype=torch.float, device=self.device
      )
      self._current_motion_failed = torch.zeros(
        num_motions, dtype=torch.float, device=self.device
      )
      self.phase_failed_count = torch.zeros(
        (num_motions, self.bin_count), dtype=torch.float, device=self.device
      )
      self._current_phase_failed = torch.zeros(
        (num_motions, self.bin_count), dtype=torch.float, device=self.device
      )
    else:
      self.bin_failed_count = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
      )
      self._current_bin_failed = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
      )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_motion_entropy"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_motion_top1_prob"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_motion_top1_idx"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_phase_entropy"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_phase_top1_prob"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_phase_top1_bin"] = torch.zeros(
      self.num_envs, device=self.device
    )

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _current_times_s(self) -> torch.Tensor:
    return (
      self.time_steps.to(torch.float32) * self._env.step_dt + self.motion_time_offsets
    )

  def _refresh_motion_frame(self) -> None:
    if not self._uses_motion_library:
      return
    assert self.motion_lib is not None
    self._current_motion_frame = self.motion_lib.calc_motion_frame(
      self.motion_ids,
      self._current_times_s(),
      anchor_body_index=self.motion_anchor_body_index,
    )

  def query_motion_frames(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
    """Query future reference frames at given step offsets."""
    if len(step_offsets) == 0:
      raise ValueError("`step_offsets` must contain at least one entry.")

    if self._uses_motion_library:
      return self._query_motion_frames_library(step_offsets)
    return self._query_motion_frames_npz(step_offsets)

  def _query_motion_frames_npz(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
    assert self.motion is not None

    offsets = torch.tensor(step_offsets, dtype=torch.long, device=self.device)
    num_steps = int(offsets.shape[0])
    frame_ids = self.time_steps[:, None] + offsets[None, :]
    frame_ids = torch.clamp(frame_ids, min=0, max=self.motion.time_step_total - 1)
    flat_frame_ids = frame_ids.reshape(-1)

    origins = (
      self._env.scene.env_origins[:, None, :].expand(-1, num_steps, -1).reshape(-1, 3)
    )

    body_pos = (
      self.motion.body_pos_w[flat_frame_ids][:, self.motion_body_indexes]
      + origins[:, None, :]
    )
    body_quat = self.motion.body_quat_w[flat_frame_ids][:, self.motion_body_indexes]
    body_lin_vel = self.motion.body_lin_vel_w[flat_frame_ids][
      :, self.motion_body_indexes
    ]
    body_ang_vel = self.motion.body_ang_vel_w[flat_frame_ids][
      :, self.motion_body_indexes
    ]

    anchor_pos = (
      self.motion.body_pos_w[flat_frame_ids, self.motion_anchor_body_index] + origins
    )
    anchor_quat = self.motion.body_quat_w[flat_frame_ids, self.motion_anchor_body_index]
    anchor_lin_vel = self.motion.body_lin_vel_w[
      flat_frame_ids, self.motion_anchor_body_index
    ]
    anchor_ang_vel = self.motion.body_ang_vel_w[
      flat_frame_ids, self.motion_anchor_body_index
    ]

    return MotionFrameBatch(
      joint_pos=self.motion.joint_pos[flat_frame_ids].reshape(
        self.num_envs, num_steps, -1
      ),
      joint_vel=self.motion.joint_vel[flat_frame_ids].reshape(
        self.num_envs, num_steps, -1
      ),
      body_pos_w=body_pos.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 3
      ),
      body_quat_w=body_quat.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 4
      ),
      body_lin_vel_w=body_lin_vel.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 3
      ),
      body_ang_vel_w=body_ang_vel.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 3
      ),
      anchor_pos_w=anchor_pos.reshape(self.num_envs, num_steps, 3),
      anchor_quat_w=anchor_quat.reshape(self.num_envs, num_steps, 4),
      anchor_lin_vel_w=anchor_lin_vel.reshape(self.num_envs, num_steps, 3),
      anchor_ang_vel_w=anchor_ang_vel.reshape(self.num_envs, num_steps, 3),
    )

  def _query_motion_frames_library(
    self, step_offsets: tuple[int, ...]
  ) -> MotionFrameBatch:
    assert self.motion_lib is not None

    offsets = torch.tensor(step_offsets, dtype=torch.float32, device=self.device)
    num_steps = int(offsets.shape[0])

    motion_ids = self.motion_ids[:, None].expand(-1, num_steps)
    query_times = (
      self._current_times_s()[:, None] + offsets[None, :] * self._env.step_dt
    )
    motion_lengths = self.motion_lib.get_motion_length(motion_ids.reshape(-1)).reshape(
      self.num_envs, num_steps
    )
    query_times = torch.clamp(query_times, min=0.0)
    query_times = torch.minimum(
      query_times, torch.clamp(motion_lengths - 1e-6, min=0.0)
    )

    flat_motion_ids = motion_ids.reshape(-1)
    flat_query_times = query_times.reshape(-1)
    flat_frames = self.motion_lib.calc_motion_frame(
      flat_motion_ids,
      flat_query_times,
      anchor_body_index=self.motion_anchor_body_index,
    )

    origins = (
      self._env.scene.env_origins[:, None, :].expand(-1, num_steps, -1).reshape(-1, 3)
    )
    body_pos = flat_frames.body_pos_w[:, self.motion_body_indexes] + origins[:, None, :]
    body_quat = flat_frames.body_quat_w[:, self.motion_body_indexes]
    body_lin_vel = flat_frames.body_lin_vel_w[:, self.motion_body_indexes]
    body_ang_vel = flat_frames.body_ang_vel_w[:, self.motion_body_indexes]
    anchor_pos = flat_frames.body_pos_w[:, self.motion_anchor_body_index] + origins
    anchor_quat = flat_frames.body_quat_w[:, self.motion_anchor_body_index]
    anchor_lin_vel = flat_frames.body_lin_vel_w[:, self.motion_anchor_body_index]
    anchor_ang_vel = flat_frames.body_ang_vel_w[:, self.motion_anchor_body_index]

    return MotionFrameBatch(
      joint_pos=flat_frames.joint_pos.reshape(self.num_envs, num_steps, -1),
      joint_vel=flat_frames.joint_vel.reshape(self.num_envs, num_steps, -1),
      body_pos_w=body_pos.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 3
      ),
      body_quat_w=body_quat.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 4
      ),
      body_lin_vel_w=body_lin_vel.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 3
      ),
      body_ang_vel_w=body_ang_vel.reshape(
        self.num_envs, num_steps, len(self.cfg.body_names), 3
      ),
      anchor_pos_w=anchor_pos.reshape(self.num_envs, num_steps, 3),
      anchor_quat_w=anchor_quat.reshape(self.num_envs, num_steps, 4),
      anchor_lin_vel_w=anchor_lin_vel.reshape(self.num_envs, num_steps, 3),
      anchor_ang_vel_w=anchor_ang_vel.reshape(self.num_envs, num_steps, 3),
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.joint_pos
    assert self.motion is not None
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.joint_vel
    assert self.motion is not None
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return (
        self._current_motion_frame.body_pos_w[:, self.motion_body_indexes]
        + self._env.scene.env_origins[:, None, :]
      )
    assert self.motion is not None
    selected = self.motion.body_pos_w[self.time_steps][:, self.motion_body_indexes]
    return selected + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_quat_w[:, self.motion_body_indexes]
    assert self.motion is not None
    return self.motion.body_quat_w[self.time_steps][:, self.motion_body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_lin_vel_w[:, self.motion_body_indexes]
    assert self.motion is not None
    return self.motion.body_lin_vel_w[self.time_steps][:, self.motion_body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_ang_vel_w[:, self.motion_body_indexes]
    assert self.motion is not None
    return self.motion.body_ang_vel_w[self.time_steps][:, self.motion_body_indexes]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return (
        self._current_motion_frame.body_pos_w[:, self.motion_anchor_body_index]
        + self._env.scene.env_origins
      )
    assert self.motion is not None
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_quat_w[:, self.motion_anchor_body_index]
    assert self.motion is not None
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_lin_vel_w[:, self.motion_anchor_body_index]
    assert self.motion is not None
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_ang_vel_w[:, self.motion_anchor_body_index]
    assert self.motion is not None
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    adaptive_sampling(self, env_ids)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    uniform_sampling(self, env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    resample_command(self, env_ids)

  def _update_command(self):
    update_command(self)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    debug_visualize_motion_command(self, visualizer, _DESIRED_FRAME_COLORS)


class JointRefMotionCommand(MotionCommand):
  """Tracking-style command representation: [joint_pos_ref, joint_vel_ref]."""


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  """Joint-ref motion command configuration (single-step command)."""

  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  motion_body_names: tuple[str, ...] | None = None
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  show_motion_load_progress: bool | None = None

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return JointRefMotionCommand(self, env)
