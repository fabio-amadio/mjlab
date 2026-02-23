from __future__ import annotations

import copy
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

try:
  from tqdm import tqdm
except ModuleNotFoundError:
  tqdm = None

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(self, motion_file: str, device: str = "cpu") -> None:
    with np.load(motion_file) as data:
      self.joint_pos = torch.tensor(
        data["joint_pos"], dtype=torch.float32, device=device
      )
      self.joint_vel = torch.tensor(
        data["joint_vel"], dtype=torch.float32, device=device
      )
      self.body_pos_w = torch.tensor(
        data["body_pos_w"], dtype=torch.float32, device=device
      )
      self.body_quat_w = torch.tensor(
        data["body_quat_w"], dtype=torch.float32, device=device
      )
      self.body_lin_vel_w = torch.tensor(
        data["body_lin_vel_w"], dtype=torch.float32, device=device
      )
      self.body_ang_vel_w = torch.tensor(
        data["body_ang_vel_w"], dtype=torch.float32, device=device
      )
      self.body_names = self._extract_body_names(data)

    self.time_step_total = self.joint_pos.shape[0]

  @staticmethod
  def _decode_name(value: object) -> str:
    if isinstance(value, bytes):
      return value.decode("utf-8")
    return str(value)

  @classmethod
  def _extract_body_names(cls, data) -> tuple[str, ...] | None:
    for key in ("body_names", "body_link_names"):
      if key in data:
        flat_values = np.asarray(data[key]).reshape(-1)
        return tuple(cls._decode_name(v) for v in flat_values.tolist())
    return None


@dataclass
class MotionFrameBatch:
  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  anchor_pos_w: torch.Tensor
  anchor_quat_w: torch.Tensor
  anchor_lin_vel_w: torch.Tensor
  anchor_ang_vel_w: torch.Tensor


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name)
  if value is None:
    return default
  try:
    return int(value)
  except ValueError:
    return default


def _should_show_progress(show_progress: bool | None) -> bool:
  if show_progress is not None:
    return show_progress
  return (
    _env_int("RANK", 0) == 0
    and _env_int("LOCAL_RANK", 0) == 0
    and sys.stderr.isatty()
  )


def _load_yaml_config(yaml_path: Path) -> dict[str, object]:
  try:
    import yaml  # type: ignore[import-not-found]
  except ModuleNotFoundError:
    return _parse_minimal_yaml(yaml_path.read_text(encoding="utf-8"), yaml_path)

  with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"YAML config must be a mapping at top level: {yaml_path}")
  return data


def _parse_yaml_scalar(value: str) -> object:
  value = value.strip()
  if value == "":
    return ""
  if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
    return value[1:-1]
  lower = value.lower()
  if lower in ("true", "false"):
    return lower == "true"
  try:
    return int(value)
  except ValueError:
    pass
  try:
    return float(value)
  except ValueError:
    pass
  return value


def _parse_minimal_yaml(text: str, yaml_path: Path) -> dict[str, object]:
  """Parse a minimal YAML subset used by CLAMP motion configs.

  Supported structure:
    root_path: ...
    subfolders:
      - name: ...
        weight: ...
  """
  cfg: dict[str, object] = {}
  active_list_key: str | None = None
  current_item: dict[str, object] | None = None

  for raw_line in text.splitlines():
    line = raw_line.split("#", 1)[0].rstrip()
    if not line.strip():
      continue
    stripped = line.lstrip()
    indent = len(line) - len(stripped)

    if indent == 0 and stripped.endswith(":"):
      key = stripped[:-1].strip()
      cfg[key] = []
      active_list_key = key
      current_item = None
      continue

    if indent == 0 and ":" in stripped:
      key, value = stripped.split(":", 1)
      cfg[key.strip()] = _parse_yaml_scalar(value)
      active_list_key = None
      current_item = None
      continue

    if active_list_key is None:
      raise ValueError(f"Unsupported YAML structure in {yaml_path}: line `{raw_line}`")

    assert isinstance(cfg[active_list_key], list)
    if stripped.startswith("-"):
      item_content = stripped[1:].strip()
      current_item = {}
      cfg[active_list_key].append(current_item)
      if item_content:
        if ":" not in item_content:
          raise ValueError(
            f"Invalid list item format in {yaml_path}: line `{raw_line}`"
          )
        key, value = item_content.split(":", 1)
        current_item[key.strip()] = _parse_yaml_scalar(value)
    else:
      if current_item is None or ":" not in stripped:
        raise ValueError(
          f"Invalid nested YAML entry in {yaml_path}: line `{raw_line}`"
        )
      key, value = stripped.split(":", 1)
      current_item[key.strip()] = _parse_yaml_scalar(value)

  return cfg


def _quat_slerp_batch(
  q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  """Vectorized quaternion slerp for tensors with matching shape [..., 4]."""
  q0 = q0 / torch.clamp(torch.norm(q0, dim=-1, keepdim=True), min=1e-8)
  q1 = q1 / torch.clamp(torch.norm(q1, dim=-1, keepdim=True), min=1e-8)
  blend = blend.unsqueeze(-1)

  dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1 = torch.where(dot < 0.0, -q1, q1)
  dot = torch.abs(dot).clamp(max=1.0)

  close = dot > 0.9995
  theta_0 = torch.acos(dot)
  sin_theta_0 = torch.sin(theta_0)
  theta = theta_0 * blend
  sin_theta = torch.sin(theta)

  s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1e-8)
  s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
  slerped = s0 * q0 + s1 * q1

  lerped = (1.0 - blend) * q0 + blend * q1
  lerped = lerped / torch.clamp(torch.norm(lerped, dim=-1, keepdim=True), min=1e-8)
  return torch.where(close, lerped, slerped)


class NpzMotionLibrary:
  """Motion library for multi-file NPZ datasets."""

  def __init__(
    self,
    motion_source: str,
    device: str = "cpu",
    show_progress: bool | None = None,
  ) -> None:
    self.device = device
    motion_files, per_motion_weights = self._resolve_motion_entries(motion_source)
    if len(motion_files) == 0:
      raise ValueError(f"No .npz motion files found in: {motion_source}")

    self.body_names: tuple[str, ...] | None = None
    self._num_dof: int | None = None
    self._num_bodies: int | None = None

    motion_num_frames: list[int] = []
    motion_lengths_s: list[float] = []
    motion_weights: list[float] = []
    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    body_pos_w_list: list[torch.Tensor] = []
    body_quat_w_list: list[torch.Tensor] = []
    body_lin_vel_w_list: list[torch.Tensor] = []
    body_ang_vel_w_list: list[torch.Tensor] = []

    for motion_file, motion_weight in self._iter_motion_entries(
      motion_files=motion_files,
      per_motion_weights=per_motion_weights,
      show_progress=show_progress,
    ):
      with np.load(motion_file) as data:
        required_keys = {
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        }
        missing_keys = required_keys.difference(set(data.keys()))
        if missing_keys:
          raise ValueError(
            "Invalid motion npz. Missing keys: "
            f"{sorted(missing_keys)} in {motion_file}"
          )

        fps = self._extract_fps(data)
        dt = 1.0 / max(fps, 1e-6)

        joint_pos = torch.tensor(
          np.asarray(data["joint_pos"]), dtype=torch.float32, device=self.device
        )
        joint_vel = torch.tensor(
          np.asarray(data["joint_vel"]), dtype=torch.float32, device=self.device
        )
        body_pos_w = torch.tensor(
          np.asarray(data["body_pos_w"]), dtype=torch.float32, device=self.device
        )
        body_quat_w = torch.tensor(
          np.asarray(data["body_quat_w"]), dtype=torch.float32, device=self.device
        )
        body_lin_vel_w = torch.tensor(
          np.asarray(data["body_lin_vel_w"]), dtype=torch.float32, device=self.device
        )
        body_ang_vel_w = torch.tensor(
          np.asarray(data["body_ang_vel_w"]), dtype=torch.float32, device=self.device
        )

        file_body_names = MotionLoader._extract_body_names(data)

      if joint_pos.ndim != 2:
        raise ValueError(f"Invalid joint_pos shape in {motion_file}: {joint_pos.shape}")
      if joint_vel.shape != joint_pos.shape:
        raise ValueError(
          f"joint_vel must match joint_pos shape in {motion_file}: "
          f"{joint_vel.shape} vs {joint_pos.shape}"
        )
      if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(f"Invalid body_pos_w shape in {motion_file}: {body_pos_w.shape}")
      if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(f"Invalid body_quat_w shape in {motion_file}: {body_quat_w.shape}")
      if body_lin_vel_w.shape != body_pos_w.shape:
        raise ValueError(
          f"body_lin_vel_w must match body_pos_w shape in {motion_file}: "
          f"{body_lin_vel_w.shape} vs {body_pos_w.shape}"
        )
      if body_ang_vel_w.shape != body_pos_w.shape:
        raise ValueError(
          f"body_ang_vel_w must match body_pos_w shape in {motion_file}: "
          f"{body_ang_vel_w.shape} vs {body_pos_w.shape}"
        )
      if (
        joint_pos.shape[0] != body_pos_w.shape[0]
        or joint_pos.shape[0] != body_quat_w.shape[0]
      ):
        raise ValueError(
          "Frame count mismatch in motion npz: "
          f"{motion_file} (joint={joint_pos.shape[0]}, body_pos={body_pos_w.shape[0]}, "
          f"body_quat={body_quat_w.shape[0]})"
        )
      if joint_pos.shape[0] < 2:
        raise ValueError(
          f"Motion {motion_file} has fewer than 2 frames: {joint_pos.shape[0]}"
        )

      if file_body_names is None:
        raise ValueError(
          "Motion npz must include body names (`body_names` or `body_link_names`): "
          f"{motion_file}"
        )
      if self.body_names is None:
        self.body_names = tuple(file_body_names)
      elif self.body_names != tuple(file_body_names):
        raise ValueError(
          "All NPZ files must share the same body name ordering. "
          f"Mismatch in {motion_file}."
        )

      if self._num_dof is None:
        self._num_dof = int(joint_pos.shape[1])
      elif self._num_dof != int(joint_pos.shape[1]):
        raise ValueError(
          "All NPZ files must share the same number of DoFs. "
          f"Expected {self._num_dof}, got {joint_pos.shape[1]} in {motion_file}."
        )

      if self._num_bodies is None:
        self._num_bodies = int(body_pos_w.shape[1])
      elif self._num_bodies != int(body_pos_w.shape[1]):
        raise ValueError(
          "All NPZ files must share the same number of bodies. "
          f"Expected {self._num_bodies}, got {body_pos_w.shape[1]} in {motion_file}."
        )

      num_frames = int(joint_pos.shape[0])
      motion_num_frames.append(num_frames)
      motion_lengths_s.append(dt * float(num_frames - 1))
      motion_weights.append(float(motion_weight))
      joint_pos_list.append(joint_pos)
      joint_vel_list.append(joint_vel)
      body_pos_w_list.append(body_pos_w)
      body_quat_w_list.append(body_quat_w)
      body_lin_vel_w_list.append(body_lin_vel_w)
      body_ang_vel_w_list.append(body_ang_vel_w)

    assert self.body_names is not None
    assert self._num_dof is not None
    assert self._num_bodies is not None

    self.motion_num_frames = torch.tensor(
      motion_num_frames, dtype=torch.long, device=self.device
    )
    self.motion_lengths_s = torch.tensor(
      motion_lengths_s, dtype=torch.float32, device=self.device
    )
    self.motion_weights = torch.tensor(
      motion_weights, dtype=torch.float32, device=self.device
    )
    if torch.all(self.motion_weights <= 0):
      self.motion_weights = torch.ones_like(self.motion_weights)
    self.motion_weights = self.motion_weights / self.motion_weights.sum()

    lengths_shifted = self.motion_num_frames.roll(1)
    lengths_shifted[0] = 0
    self.motion_start_idx = lengths_shifted.cumsum(0)

    self.joint_pos = torch.cat(joint_pos_list, dim=0)
    self.joint_vel = torch.cat(joint_vel_list, dim=0)
    self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
    self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
    self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
    self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)

  @staticmethod
  def _extract_fps(data) -> float:
    if "fps" not in data:
      return 30.0
    fps_value = np.asarray(data["fps"]).reshape(-1)
    if fps_value.size == 0:
      return 30.0
    return float(fps_value[0])

  @classmethod
  def _iter_motion_entries(
    cls,
    motion_files: list[Path],
    per_motion_weights: list[float],
    show_progress: bool | None,
  ):
    entries = list(zip(motion_files, per_motion_weights))
    if _should_show_progress(show_progress) and tqdm is not None and len(entries) > 1:
      return tqdm(
        entries,
        total=len(entries),
        desc="Loading motion NPZs",
        unit="file",
        leave=False,
        dynamic_ncols=True,
      )
    return entries

  @classmethod
  def _resolve_motion_entries(
    cls, motion_source: str
  ) -> tuple[list[Path], list[float]]:
    source = Path(motion_source)
    if source.suffix in (".yaml", ".yml") and source.is_file():
      return cls._resolve_motion_entries_from_yaml(source)
    if source.is_dir():
      files = sorted(source.rglob("*.npz"))
      return files, [1.0] * len(files)
    if source.suffix == ".npz" and source.is_file():
      return [source], [1.0]
    raise ValueError(
      "NPZ motion source must be an existing .npz/.yaml file or directory. "
      f"Got: {motion_source}"
    )

  @classmethod
  def _resolve_motion_entries_from_yaml(
    cls, yaml_path: Path
  ) -> tuple[list[Path], list[float]]:
    config = _load_yaml_config(yaml_path)
    root_path_raw = config.get("root_path", ".")
    if not isinstance(root_path_raw, str):
      raise ValueError(
        f"`root_path` must be a string in {yaml_path}. Got: {type(root_path_raw)}"
      )
    root_path = Path(root_path_raw)
    if not root_path.is_absolute():
      root_path = (yaml_path.parent / root_path).resolve()

    subfolders = config.get("subfolders")
    if not isinstance(subfolders, list):
      raise ValueError(
        f"`subfolders` must be a list in {yaml_path}. "
        "Expected entries like `{name: cnrs, weight: 1.0}`."
      )

    motion_files: list[Path] = []
    motion_weights: list[float] = []
    for entry in subfolders:
      if not isinstance(entry, dict):
        raise ValueError(f"Invalid subfolder entry in {yaml_path}: {entry}")

      folder_name = entry.get("name", entry.get("folder", entry.get("subfolder")))
      if not isinstance(folder_name, str) or folder_name == "":
        raise ValueError(
          f"Subfolder entry is missing `name` (or `folder`) in {yaml_path}: {entry}"
        )

      weight_raw = entry.get("weight", 1.0)
      try:
        weight = float(weight_raw)
      except (TypeError, ValueError) as exc:
        raise ValueError(
          f"Invalid weight for subfolder `{folder_name}` in {yaml_path}: {weight_raw}"
        ) from exc
      if weight < 0.0:
        raise ValueError(
          f"Weight for subfolder `{folder_name}` must be non-negative in {yaml_path}."
        )

      folder_path = root_path / folder_name
      if not folder_path.exists():
        raise ValueError(
          f"Configured subfolder does not exist: {folder_path} (from {yaml_path})"
        )

      folder_files = sorted(folder_path.rglob("*.npz"))
      if len(folder_files) == 0:
        warnings.warn(
          f"No .npz motions found in configured subfolder: {folder_path}",
          stacklevel=2,
        )
        continue

      motion_files.extend(folder_files)
      motion_weights.extend([weight] * len(folder_files))

    if len(motion_files) == 0:
      raise ValueError(f"No .npz files resolved from YAML config: {yaml_path}")
    return motion_files, motion_weights

  def num_motions(self) -> int:
    return int(self.motion_num_frames.shape[0])

  def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return self.motion_lengths_s[motion_ids]

  def sample_motions(self, n: int) -> torch.Tensor:
    return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

  def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
    return torch.rand(motion_ids.shape, device=self.device) * self.get_motion_length(
      motion_ids
    )

  def _calc_frame_blend(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths_s = self.get_motion_length(motion_ids)
    motion_times = torch.clamp(motion_times, min=0.0)
    motion_times = torch.minimum(motion_times, torch.clamp(lengths_s - 1e-6, min=0.0))

    num_frames = self.motion_num_frames[motion_ids]
    phase = motion_times / torch.clamp(lengths_s, min=1e-6)
    phase = torch.clamp(phase, 0.0, 1.0)

    frame_idx0 = (phase * (num_frames - 1).float()).long()
    frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
    blend = phase * (num_frames - 1).float() - frame_idx0.float()

    start_idx = self.motion_start_idx[motion_ids]
    frame_idx0 = frame_idx0 + start_idx
    frame_idx1 = frame_idx1 + start_idx
    return frame_idx0, frame_idx1, blend

  def calc_motion_frame(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> MotionFrameBatch:
    frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

    joint_pos0 = self.joint_pos[frame_idx0]
    joint_pos1 = self.joint_pos[frame_idx1]
    joint_vel = self.joint_vel[frame_idx0]

    body_pos0 = self.body_pos_w[frame_idx0]
    body_pos1 = self.body_pos_w[frame_idx1]
    body_quat0 = self.body_quat_w[frame_idx0]
    body_quat1 = self.body_quat_w[frame_idx1]
    body_lin_vel = self.body_lin_vel_w[frame_idx0]
    body_ang_vel = self.body_ang_vel_w[frame_idx0]

    blend_joint = blend.unsqueeze(-1)
    blend_body = blend.unsqueeze(-1).unsqueeze(-1)
    joint_pos = (1.0 - blend_joint) * joint_pos0 + blend_joint * joint_pos1
    body_pos_w = (1.0 - blend_body) * body_pos0 + blend_body * body_pos1

    num_bodies = body_pos_w.shape[1]
    body_quat_w = _quat_slerp_batch(
      body_quat0.reshape(-1, 4),
      body_quat1.reshape(-1, 4),
      blend.unsqueeze(-1).expand(-1, num_bodies).reshape(-1),
    ).reshape(motion_ids.shape[0], num_bodies, 4)

    return MotionFrameBatch(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel,
      body_ang_vel_w=body_ang_vel,
      anchor_pos_w=body_pos_w[:, 0],
      anchor_quat_w=body_quat_w[:, 0],
      anchor_lin_vel_w=body_lin_vel[:, 0],
      anchor_ang_vel_w=body_ang_vel[:, 0],
    )


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

    motion_body_names = self._resolve_motion_body_names()
    motion_name_to_index = self._build_name_to_index(motion_body_names, source="motion")
    robot_body_names = tuple(self.robot.body_names)
    robot_name_to_index = self._build_name_to_index(robot_body_names, source="robot")

    required_body_names = list(dict.fromkeys((self.cfg.anchor_body_name, *cfg.body_names)))
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

  def _resolve_motion_body_names(self) -> tuple[str, ...]:
    """Resolve body names for the motion tensors.

    Priority:
    1) Names embedded in the motion file (`body_names`/`body_link_names`).
    2) Names explicitly provided in the config (`motion_body_names`).
    3) Fallback to robot body names if tensor count matches exactly.
    """
    if self._uses_motion_library:
      assert self.motion_lib is not None
      assert self.motion_lib.body_names is not None
      motion_tensor_body_count = len(self.motion_lib.body_names)
      file_motion_body_names = self.motion_lib.body_names
    else:
      assert self.motion is not None
      motion_tensor_body_count = int(self.motion.body_pos_w.shape[1])
      file_motion_body_names = self.motion.body_names
    cfg_motion_body_names = self.cfg.motion_body_names

    if file_motion_body_names is not None:
      if len(file_motion_body_names) != motion_tensor_body_count:
        raise ValueError(
          "Motion file body name count does not match body tensor shape: "
          f"names={len(file_motion_body_names)} tensor_bodies={motion_tensor_body_count}"
        )
      if cfg_motion_body_names is not None and tuple(cfg_motion_body_names) != tuple(
        file_motion_body_names
      ):
        raise ValueError(
          "Both motion file and cfg.motion_body_names are provided but differ. "
          "Please keep only one source of truth or make them identical."
        )
      return tuple(file_motion_body_names)

    if cfg_motion_body_names is not None:
      if len(cfg_motion_body_names) != motion_tensor_body_count:
        raise ValueError(
          "cfg.motion_body_names count does not match motion body tensor shape: "
          f"names={len(cfg_motion_body_names)} tensor_bodies={motion_tensor_body_count}"
        )
      return tuple(cfg_motion_body_names)

    robot_body_names = tuple(self.robot.body_names)
    if motion_tensor_body_count == len(robot_body_names):
      warnings.warn(
        "Motion file has no body names and cfg.motion_body_names is unset. "
        "Falling back to robot body names because counts match.",
        stacklevel=2,
      )
      return robot_body_names

    raise ValueError(
      "Unable to resolve motion body names. Provide names in the motion source "
      "or set cfg.motion_body_names."
    )

  @staticmethod
  def _build_name_to_index(
    body_names: tuple[str, ...], source: str
  ) -> dict[str, int]:
    name_to_index: dict[str, int] = {}
    duplicates: list[str] = []
    for index, name in enumerate(body_names):
      if name in name_to_index:
        duplicates.append(name)
      else:
        name_to_index[name] = index
    if duplicates:
      raise ValueError(
        f"Duplicate body names found in {source} definition: {sorted(set(duplicates))}"
      )
    return name_to_index

  def _current_times_s(self) -> torch.Tensor:
    return self.time_steps.to(torch.float32) * self._env.step_dt + self.motion_time_offsets

  def _refresh_motion_frame(self) -> None:
    if not self._uses_motion_library:
      return
    assert self.motion_lib is not None
    self._current_motion_frame = self.motion_lib.calc_motion_frame(
      self.motion_ids, self._current_times_s()
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

    origins = self._env.scene.env_origins[:, None, :].expand(-1, num_steps, -1).reshape(-1, 3)

    body_pos = self.motion.body_pos_w[flat_frame_ids][:, self.motion_body_indexes] + origins[:, None, :]
    body_quat = self.motion.body_quat_w[flat_frame_ids][:, self.motion_body_indexes]
    body_lin_vel = self.motion.body_lin_vel_w[flat_frame_ids][:, self.motion_body_indexes]
    body_ang_vel = self.motion.body_ang_vel_w[flat_frame_ids][:, self.motion_body_indexes]

    anchor_pos = self.motion.body_pos_w[flat_frame_ids, self.motion_anchor_body_index] + origins
    anchor_quat = self.motion.body_quat_w[flat_frame_ids, self.motion_anchor_body_index]
    anchor_lin_vel = self.motion.body_lin_vel_w[flat_frame_ids, self.motion_anchor_body_index]
    anchor_ang_vel = self.motion.body_ang_vel_w[flat_frame_ids, self.motion_anchor_body_index]

    return MotionFrameBatch(
      joint_pos=self.motion.joint_pos[flat_frame_ids].reshape(self.num_envs, num_steps, -1),
      joint_vel=self.motion.joint_vel[flat_frame_ids].reshape(self.num_envs, num_steps, -1),
      body_pos_w=body_pos.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 3),
      body_quat_w=body_quat.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 4),
      body_lin_vel_w=body_lin_vel.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 3),
      body_ang_vel_w=body_ang_vel.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 3),
      anchor_pos_w=anchor_pos.reshape(self.num_envs, num_steps, 3),
      anchor_quat_w=anchor_quat.reshape(self.num_envs, num_steps, 4),
      anchor_lin_vel_w=anchor_lin_vel.reshape(self.num_envs, num_steps, 3),
      anchor_ang_vel_w=anchor_ang_vel.reshape(self.num_envs, num_steps, 3),
    )

  def _query_motion_frames_library(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
    assert self.motion_lib is not None

    offsets = torch.tensor(step_offsets, dtype=torch.float32, device=self.device)
    num_steps = int(offsets.shape[0])

    motion_ids = self.motion_ids[:, None].expand(-1, num_steps)
    query_times = self._current_times_s()[:, None] + offsets[None, :] * self._env.step_dt
    motion_lengths = self.motion_lib.get_motion_length(motion_ids.reshape(-1)).reshape(
      self.num_envs, num_steps
    )
    query_times = torch.clamp(query_times, min=0.0)
    query_times = torch.minimum(query_times, torch.clamp(motion_lengths - 1e-6, min=0.0))

    flat_motion_ids = motion_ids.reshape(-1)
    flat_query_times = query_times.reshape(-1)
    flat_frames = self.motion_lib.calc_motion_frame(flat_motion_ids, flat_query_times)

    origins = self._env.scene.env_origins[:, None, :].expand(-1, num_steps, -1).reshape(-1, 3)
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
      body_pos_w=body_pos.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 3),
      body_quat_w=body_quat.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 4),
      body_lin_vel_w=body_lin_vel.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 3),
      body_ang_vel_w=body_ang_vel.reshape(self.num_envs, num_steps, len(self.cfg.body_names), 3),
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
    if self._uses_motion_library:
      assert self.motion_lib is not None
      assert self.motion_failed_count is not None
      assert self._current_motion_failed is not None
      assert self.phase_failed_count is not None
      assert self._current_phase_failed is not None

      self._current_motion_failed.zero_()
      self._current_phase_failed.zero_()

      episode_failed = self._env.termination_manager.terminated[env_ids]
      if torch.any(episode_failed):
        failed_env_ids = env_ids[episode_failed]
        failed_motion_ids = self.motion_ids[failed_env_ids]
        failed_times = self._current_times_s()[failed_env_ids]
        failed_lengths = self.motion_lib.get_motion_length(failed_motion_ids)
        failed_phase_bins = torch.clamp(
          (failed_times / torch.clamp(failed_lengths, min=1.0e-6) * self.bin_count).long(),
          0,
          self.bin_count - 1,
        )
        self._current_motion_failed[:] = torch.bincount(
          failed_motion_ids, minlength=self.motion_lib.num_motions()
        ).to(dtype=torch.float32)
        phase_flat_idx = failed_motion_ids * self.bin_count + failed_phase_bins
        self._current_phase_failed[:] = torch.bincount(
          phase_flat_idx, minlength=self.motion_lib.num_motions() * self.bin_count
        ).to(dtype=torch.float32).reshape(self.motion_lib.num_motions(), self.bin_count)

      mix_ratio = float(min(max(self.cfg.adaptive_uniform_ratio, 0.0), 1.0))
      hard_motion_prob = self.motion_failed_count + 1.0 / float(self.motion_lib.num_motions())
      hard_motion_prob = hard_motion_prob / hard_motion_prob.sum()
      motion_probabilities = (
        (1.0 - mix_ratio) * hard_motion_prob + mix_ratio * self.motion_lib.motion_weights
      )
      motion_probabilities = motion_probabilities / motion_probabilities.sum()

      hard_phase_prob = self.phase_failed_count + 1.0 / float(self.bin_count)
      hard_phase_prob = torch.nn.functional.pad(
        hard_phase_prob.unsqueeze(1),
        (0, self.cfg.adaptive_kernel_size - 1),
        mode="replicate",
      )
      hard_phase_prob = torch.nn.functional.conv1d(
        hard_phase_prob, self.kernel.view(1, 1, -1)
      ).squeeze(1)
      hard_phase_prob = hard_phase_prob / torch.clamp(
        hard_phase_prob.sum(dim=1, keepdim=True), min=1.0e-8
      )

      uniform_phase_prob = torch.full_like(hard_phase_prob, 1.0 / float(self.bin_count))
      phase_probabilities = (
        (1.0 - mix_ratio) * hard_phase_prob + mix_ratio * uniform_phase_prob
      )
      phase_probabilities = phase_probabilities / torch.clamp(
        phase_probabilities.sum(dim=1, keepdim=True), min=1.0e-8
      )

      sampled_motion_ids = torch.multinomial(
        motion_probabilities, len(env_ids), replacement=True
      )
      sampled_phase_prob = phase_probabilities[sampled_motion_ids]
      sampled_bins = torch.multinomial(sampled_phase_prob, 1, replacement=True).squeeze(-1)
      sampled_phase = (
        sampled_bins
        + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
      ) / self.bin_count
      sampled_lengths = self.motion_lib.get_motion_length(sampled_motion_ids)
      sampled_offsets = sampled_phase * sampled_lengths
      sampled_offsets = torch.minimum(
        sampled_offsets,
        torch.clamp(sampled_lengths - 1.0e-6, min=0.0),
      )

      self.motion_ids[env_ids] = sampled_motion_ids
      self.motion_time_offsets[env_ids] = sampled_offsets
      self.time_steps[env_ids] = 0
      self._refresh_motion_frame()

      motion_top_prob, motion_top_idx = motion_probabilities.max(dim=0)
      phase_top_prob, phase_top_idx = phase_probabilities[motion_top_idx].max(dim=0)
      if self.motion_lib.num_motions() > 1:
        motion_entropy = -(
          motion_probabilities * (motion_probabilities + 1.0e-12).log()
        ).sum() / math.log(self.motion_lib.num_motions())
      else:
        motion_entropy = torch.tensor(1.0, device=self.device)
      if self.bin_count > 1:
        phase_entropy = -(
          phase_probabilities * (phase_probabilities + 1.0e-12).log()
        ).sum(dim=1).mean() / math.log(self.bin_count)
      else:
        phase_entropy = torch.tensor(1.0, device=self.device)

      self.metrics["sampling_motion_entropy"][:] = motion_entropy
      self.metrics["sampling_motion_top1_prob"][:] = motion_top_prob
      self.metrics["sampling_motion_top1_idx"][:] = (
        motion_top_idx.float() / self.motion_lib.num_motions()
      )
      self.metrics["sampling_phase_entropy"][:] = phase_entropy
      self.metrics["sampling_phase_top1_prob"][:] = phase_top_prob
      self.metrics["sampling_phase_top1_bin"][:] = phase_top_idx.float() / self.bin_count
      self.metrics["sampling_entropy"][:] = 0.5 * (motion_entropy + phase_entropy)
      self.metrics["sampling_top1_prob"][:] = motion_top_prob * phase_top_prob
      self.metrics["sampling_top1_bin"][:] = phase_top_idx.float() / self.bin_count
      return

    assert self.motion is not None
    assert self.bin_failed_count is not None
    assert self._current_bin_failed is not None
    episode_failed = self._env.termination_manager.terminated[env_ids]
    self._current_bin_failed.zero_()
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count)
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_motion_entropy"][:] = 0.0
    self.metrics["sampling_motion_top1_prob"][:] = 0.0
    self.metrics["sampling_motion_top1_idx"][:] = 0.0
    self.metrics["sampling_phase_entropy"][:] = H_norm
    self.metrics["sampling_phase_top1_prob"][:] = pmax
    self.metrics["sampling_phase_top1_bin"][:] = imax.float() / self.bin_count
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    if self._uses_motion_library:
      assert self.motion_lib is not None
      sampled_motion_ids = self.motion_lib.sample_motions(len(env_ids))
      self.motion_ids[env_ids] = sampled_motion_ids
      self.motion_time_offsets[env_ids] = self.motion_lib.sample_time(sampled_motion_ids)
      self.time_steps[env_ids] = 0
      self._refresh_motion_frame()

      motion_probabilities = self.motion_lib.motion_weights
      motion_top_prob, motion_top_idx = motion_probabilities.max(dim=0)
      if self.motion_lib.num_motions() > 1:
        motion_entropy = -(
          motion_probabilities * (motion_probabilities + 1.0e-12).log()
        ).sum() / math.log(self.motion_lib.num_motions())
      else:
        motion_entropy = torch.tensor(1.0, device=self.device)
      self.metrics["sampling_motion_entropy"][:] = motion_entropy
      self.metrics["sampling_motion_top1_prob"][:] = motion_top_prob
      self.metrics["sampling_motion_top1_idx"][:] = (
        motion_top_idx.float() / self.motion_lib.num_motions()
      )
      self.metrics["sampling_phase_entropy"][:] = 1.0
      self.metrics["sampling_phase_top1_prob"][:] = 1.0 / self.bin_count
      self.metrics["sampling_phase_top1_bin"][:] = 0.5
    else:
      assert self.motion is not None
      self.time_steps[env_ids] = torch.randint(
        0, self.motion.time_step_total, (len(env_ids),), device=self.device
      )
      self.metrics["sampling_motion_entropy"][:] = 0.0
      self.metrics["sampling_motion_top1_prob"][:] = 0.0
      self.metrics["sampling_motion_top1_idx"][:] = 0.0
      self.metrics["sampling_phase_entropy"][:] = 1.0
      self.metrics["sampling_phase_top1_prob"][:] = 1.0 / self.bin_count
      self.metrics["sampling_phase_top1_bin"][:] = 0.5

    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      if self._uses_motion_library:
        assert self.motion_lib is not None
        self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
        self.motion_time_offsets[env_ids] = 0.0
      self.time_steps[env_ids] = 0
      if self._uses_motion_library:
        self._refresh_motion_frame()
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.anchor_pos_w.clone()
    root_ori = self.anchor_quat_w.clone()
    root_lin_vel = self.anchor_lin_vel_w.clone()
    root_ang_vel = self.anchor_ang_vel_w.clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore[arg-type]
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1

    if self._uses_motion_library:
      assert self.motion_lib is not None
      motion_times = self._current_times_s()
      motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
      env_ids = torch.where(motion_times >= motion_lengths)[0]
      if env_ids.numel() > 0:
        self._resample_command(env_ids)
      self._refresh_motion_frame()
    else:
      assert self.motion is not None
      env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
      if env_ids.numel() > 0:
        self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      if self._uses_motion_library:
        assert self.motion_failed_count is not None
        assert self._current_motion_failed is not None
        assert self.phase_failed_count is not None
        assert self._current_phase_failed is not None
        self.motion_failed_count = (
          self.cfg.adaptive_alpha * self._current_motion_failed
          + (1 - self.cfg.adaptive_alpha) * self.motion_failed_count
        )
        self.phase_failed_count = (
          self.cfg.adaptive_alpha * self._current_phase_failed
          + (1 - self.cfg.adaptive_alpha) * self.phase_failed_count
        )
        self._current_motion_failed.zero_()
        self._current_phase_failed.zero_()
      else:
        assert self.bin_failed_count is not None
        assert self._current_bin_failed is not None
        self.bin_failed_count = (
          self.cfg.adaptive_alpha * self._current_bin_failed
          + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.anchor_pos_w[batch].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.anchor_quat_w[batch].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()
        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
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
    return MotionCommand(self, env)
