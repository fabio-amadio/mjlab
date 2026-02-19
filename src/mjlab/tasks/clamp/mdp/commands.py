from __future__ import annotations

import copy
import math
import os
import pickle
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
  axis_angle_from_quat,
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
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
    with np.load(motion_file, allow_pickle=True) as data:
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


class PklMotionLibrary:
  """Motion library for TWIST-style PKL files.

  Supports loading a single PKL file or recursively scanning a directory.
  """

  def __init__(
    self,
    motion_source: str,
    device: str = "cpu",
    show_progress: bool | None = None,
  ) -> None:
    self.device = device
    motion_files, per_motion_weights = self._resolve_motion_entries(motion_source)
    if len(motion_files) == 0:
      raise ValueError(f"No .pkl motion files found in: {motion_source}")

    self.body_names: tuple[str, ...] | None = None
    self._num_dof: int | None = None
    self._has_local_body_rot = True

    motion_num_frames: list[int] = []
    motion_lengths_s: list[float] = []
    motion_weights: list[float] = []
    root_pos_list: list[torch.Tensor] = []
    root_rot_list: list[torch.Tensor] = []
    root_vel_list: list[torch.Tensor] = []
    root_ang_vel_list: list[torch.Tensor] = []
    dof_pos_list: list[torch.Tensor] = []
    dof_vel_list: list[torch.Tensor] = []
    local_body_pos_list: list[torch.Tensor] = []
    local_body_lin_vel_list: list[torch.Tensor] = []
    local_body_rot_list: list[torch.Tensor] = []
    local_body_ang_vel_list: list[torch.Tensor] = []

    for motion_file, motion_weight in self._iter_motion_entries(
      motion_files=motion_files,
      per_motion_weights=per_motion_weights,
      show_progress=show_progress,
    ):
      with open(motion_file, "rb") as f:
        motion_data = pickle.load(f)

      required_keys = {"root_pos", "root_rot", "dof_pos", "local_body_pos", "link_body_list"}
      missing_keys = required_keys.difference(set(motion_data.keys()))
      if missing_keys:
        raise ValueError(
          "Invalid motion pkl. Missing keys: "
          f"{sorted(missing_keys)} in {motion_file}"
        )

      fps = float(motion_data.get("fps", 30.0))
      dt = 1.0 / max(fps, 1e-6)

      root_pos = torch.tensor(
        np.asarray(motion_data["root_pos"]), dtype=torch.float32, device=self.device
      )
      root_rot = torch.tensor(
        np.asarray(motion_data["root_rot"]), dtype=torch.float32, device=self.device
      )
      dof_pos = torch.tensor(
        np.asarray(motion_data["dof_pos"]), dtype=torch.float32, device=self.device
      )
      local_body_pos = torch.tensor(
        np.asarray(motion_data["local_body_pos"]), dtype=torch.float32, device=self.device
      )

      if root_pos.shape[0] < 2:
        raise ValueError(
          f"Motion {motion_file} has fewer than 2 frames: {root_pos.shape[0]}"
        )

      link_body_list = tuple(
        MotionLoader._decode_name(name) for name in list(motion_data["link_body_list"])
      )
      if self.body_names is None:
        self.body_names = link_body_list
      elif self.body_names != link_body_list:
        raise ValueError(
          "All PKL files must share the same `link_body_list` order. "
          f"Mismatch in {motion_file}."
        )

      if self._num_dof is None:
        self._num_dof = int(dof_pos.shape[1])
      elif self._num_dof != int(dof_pos.shape[1]):
        raise ValueError(
          "All PKL files must share the same number of DoFs. "
          f"Expected {self._num_dof}, got {dof_pos.shape[1]} in {motion_file}."
        )

      root_vel = self._finite_difference(root_pos, dt)
      root_ang_vel = self._quat_angular_velocity(root_rot, dt)
      dof_vel = self._finite_difference(dof_pos, dt)
      local_body_lin_vel = self._finite_difference(local_body_pos, dt)

      local_body_rot = None
      local_body_ang_vel = None
      if "local_body_rot" in motion_data:
        local_body_rot = torch.tensor(
          np.asarray(motion_data["local_body_rot"]),
          dtype=torch.float32,
          device=self.device,
        )
        local_body_ang_vel = self._quat_angular_velocity(local_body_rot, dt)
      else:
        self._has_local_body_rot = False

      num_frames = int(root_pos.shape[0])
      motion_num_frames.append(num_frames)
      motion_lengths_s.append(dt * float(num_frames - 1))
      motion_weights.append(float(motion_weight))
      root_pos_list.append(root_pos)
      root_rot_list.append(root_rot)
      root_vel_list.append(root_vel)
      root_ang_vel_list.append(root_ang_vel)
      dof_pos_list.append(dof_pos)
      dof_vel_list.append(dof_vel)
      local_body_pos_list.append(local_body_pos)
      local_body_lin_vel_list.append(local_body_lin_vel)
      if local_body_rot is not None and local_body_ang_vel is not None:
        local_body_rot_list.append(local_body_rot)
        local_body_ang_vel_list.append(local_body_ang_vel)

    assert self.body_names is not None
    assert self._num_dof is not None

    self.motion_num_frames = torch.tensor(
      motion_num_frames, dtype=torch.long, device=self.device
    )
    self.motion_lengths_s = torch.tensor(
      motion_lengths_s, dtype=torch.float32, device=self.device
    )
    self.motion_weights = torch.tensor(
      motion_weights, dtype=torch.float32, device=self.device
    )
    self.motion_weights = self.motion_weights / self.motion_weights.sum()

    lengths_shifted = self.motion_num_frames.roll(1)
    lengths_shifted[0] = 0
    self.motion_start_idx = lengths_shifted.cumsum(0)

    self.root_pos = torch.cat(root_pos_list, dim=0)
    self.root_rot = torch.cat(root_rot_list, dim=0)
    self.root_vel = torch.cat(root_vel_list, dim=0)
    self.root_ang_vel = torch.cat(root_ang_vel_list, dim=0)
    self.dof_pos = torch.cat(dof_pos_list, dim=0)
    self.dof_vel = torch.cat(dof_vel_list, dim=0)
    self.local_body_pos = torch.cat(local_body_pos_list, dim=0)
    self.local_body_lin_vel = torch.cat(local_body_lin_vel_list, dim=0)

    if self._has_local_body_rot and len(local_body_rot_list) > 0:
      self.local_body_rot = torch.cat(local_body_rot_list, dim=0)
      self.local_body_ang_vel = torch.cat(local_body_ang_vel_list, dim=0)
    else:
      self.local_body_rot = None
      self.local_body_ang_vel = None
      warnings.warn(
        "PKL motion data has no `local_body_rot`; body orientations will use root rotation only.",
        stacklevel=2,
      )

  @staticmethod
  def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
      return default
    try:
      return int(value)
    except ValueError:
      return default

  @classmethod
  def _should_show_progress(cls, show_progress: bool | None) -> bool:
    if show_progress is not None:
      return show_progress
    return (
      cls._env_int("RANK", 0) == 0
      and cls._env_int("LOCAL_RANK", 0) == 0
      and sys.stderr.isatty()
    )

  @classmethod
  def _iter_motion_entries(
    cls,
    motion_files: list[Path],
    per_motion_weights: list[float],
    show_progress: bool | None,
  ):
    entries = list(zip(motion_files, per_motion_weights))
    if (
      cls._should_show_progress(show_progress)
      and tqdm is not None
      and len(entries) > 1
    ):
      return tqdm(
        entries,
        total=len(entries),
        desc="Loading motion PKLs",
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
      files = sorted(source.rglob("*.pkl"))
      return files, [1.0] * len(files)
    if source.suffix == ".pkl" and source.is_file():
      return [source], [1.0]
    raise ValueError(
      "PKL motion source must be an existing .pkl/.yaml file or directory. "
      f"Got: {motion_source}"
    )

  @classmethod
  def _resolve_motion_entries_from_yaml(
    cls, yaml_path: Path
  ) -> tuple[list[Path], list[float]]:
    config = cls._load_yaml_config(yaml_path)
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

      folder_files = sorted(folder_path.rglob("*.pkl"))
      if len(folder_files) == 0:
        warnings.warn(
          f"No .pkl motions found in configured subfolder: {folder_path}",
          stacklevel=2,
        )
        continue

      motion_files.extend(folder_files)
      motion_weights.extend([weight] * len(folder_files))

    if len(motion_files) == 0:
      raise ValueError(f"No .pkl files resolved from YAML config: {yaml_path}")
    return motion_files, motion_weights

  @classmethod
  def _load_yaml_config(cls, yaml_path: Path) -> dict[str, object]:
    try:
      import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
      return cls._parse_minimal_yaml(yaml_path.read_text(encoding="utf-8"), yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f)
    if not isinstance(data, dict):
      raise ValueError(f"YAML config must be a mapping at top level: {yaml_path}")
    return data

  @staticmethod
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

  @classmethod
  def _parse_minimal_yaml(cls, text: str, yaml_path: Path) -> dict[str, object]:
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
        cfg[key.strip()] = cls._parse_yaml_scalar(value)
        active_list_key = None
        current_item = None
        continue

      if active_list_key is None:
        raise ValueError(
          f"Unsupported YAML structure in {yaml_path}: line `{raw_line}`"
        )

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
          current_item[key.strip()] = cls._parse_yaml_scalar(value)
      else:
        if current_item is None or ":" not in stripped:
          raise ValueError(
            f"Invalid nested YAML entry in {yaml_path}: line `{raw_line}`"
          )
        key, value = stripped.split(":", 1)
        current_item[key.strip()] = cls._parse_yaml_scalar(value)

    return cfg

  @staticmethod
  def _finite_difference(values: torch.Tensor, dt: float) -> torch.Tensor:
    vel = torch.zeros_like(values)
    if values.shape[0] > 1:
      vel[:-1] = (values[1:] - values[:-1]) / max(dt, 1e-6)
      vel[-1] = vel[-2]
    return vel

  @staticmethod
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

  @classmethod
  def _quat_angular_velocity(cls, quats: torch.Tensor, dt: float) -> torch.Tensor:
    vel = torch.zeros((*quats.shape[:-1], 3), dtype=quats.dtype, device=quats.device)
    if quats.shape[0] > 1:
      q_prev = quats[:-1]
      q_next = quats[1:]
      q_rel = quat_mul(q_next, quat_conjugate(q_prev))
      vel[:-1] = axis_angle_from_quat(q_rel) / max(dt, 1e-6)
      vel[-1] = vel[-2]
    return vel

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
    # Clamp to avoid querying past the last valid interpolation segment.
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

    root_pos0 = self.root_pos[frame_idx0]
    root_pos1 = self.root_pos[frame_idx1]
    root_rot0 = self.root_rot[frame_idx0]
    root_rot1 = self.root_rot[frame_idx1]
    root_vel = self.root_vel[frame_idx0]
    root_ang_vel = self.root_ang_vel[frame_idx0]

    dof_pos0 = self.dof_pos[frame_idx0]
    dof_pos1 = self.dof_pos[frame_idx1]
    dof_vel = self.dof_vel[frame_idx0]

    local_body_pos0 = self.local_body_pos[frame_idx0]
    local_body_pos1 = self.local_body_pos[frame_idx1]
    local_body_lin_vel = self.local_body_lin_vel[frame_idx0]

    blend_body = blend.unsqueeze(-1).unsqueeze(-1)
    root_pos = (1.0 - blend.unsqueeze(-1)) * root_pos0 + blend.unsqueeze(-1) * root_pos1
    root_rot = self._quat_slerp_batch(root_rot0, root_rot1, blend)
    dof_pos = (1.0 - blend.unsqueeze(-1)) * dof_pos0 + blend.unsqueeze(-1) * dof_pos1
    local_body_pos = (1.0 - blend_body) * local_body_pos0 + blend_body * local_body_pos1

    num_bodies = local_body_pos.shape[1]
    root_rot_expand = root_rot.unsqueeze(1).expand(-1, num_bodies, -1)
    rel_body_pos_w = quat_apply(root_rot_expand, local_body_pos)
    body_pos_w = root_pos.unsqueeze(1) + rel_body_pos_w

    if self.local_body_rot is not None and self.local_body_ang_vel is not None:
      local_body_rot0 = self.local_body_rot[frame_idx0]
      local_body_rot1 = self.local_body_rot[frame_idx1]
      flat_rot0 = local_body_rot0.reshape(-1, 4)
      flat_rot1 = local_body_rot1.reshape(-1, 4)
      flat_blend = blend.unsqueeze(-1).expand(-1, num_bodies).reshape(-1)
      local_body_rot = self._quat_slerp_batch(flat_rot0, flat_rot1, flat_blend).reshape(
        motion_ids.shape[0], num_bodies, 4
      )
      body_quat_w = quat_mul(root_rot_expand, local_body_rot)
      body_ang_vel_w = root_ang_vel.unsqueeze(1) + quat_apply(
        root_rot_expand, self.local_body_ang_vel[frame_idx0]
      )
    else:
      body_quat_w = root_rot_expand
      body_ang_vel_w = root_ang_vel.unsqueeze(1).expand(-1, num_bodies, -1)

    body_lin_vel_w = (
      root_vel.unsqueeze(1)
      + quat_apply(root_rot_expand, local_body_lin_vel)
      + torch.cross(root_ang_vel.unsqueeze(1).expand(-1, num_bodies, -1), rel_body_pos_w, dim=-1)
    )

    return MotionFrameBatch(
      joint_pos=dof_pos,
      joint_vel=dof_vel,
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel_w,
      body_ang_vel_w=body_ang_vel_w,
      anchor_pos_w=root_pos,
      anchor_quat_w=root_rot,
      anchor_lin_vel_w=root_vel,
      anchor_ang_vel_w=root_ang_vel,
    )

class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._uses_pkl_motion = self._is_pkl_motion_source(self.cfg.motion_file)

    self.motion: MotionLoader | None = None
    self.motion_lib: PklMotionLibrary | None = None
    self._current_motion_frame: MotionFrameBatch | None = None

    if self._uses_pkl_motion:
      self.motion_lib = PklMotionLibrary(
        self.cfg.motion_file,
        device=self.device,
        show_progress=self.cfg.show_motion_load_progress,
      )
    else:
      self.motion = MotionLoader(self.cfg.motion_file, device=self.device)

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

    if self._uses_pkl_motion:
      assert self.motion_lib is not None
      max_motion_len_s = float(torch.max(self.motion_lib.motion_lengths_s).item())
      self.bin_count = max(int(max_motion_len_s / max(env.step_dt, 1e-6)) + 1, 1)
      self._refresh_motion_frame()
    else:
      assert self.motion is not None
      self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1

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

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @staticmethod
  def _is_pkl_motion_source(motion_source: str) -> bool:
    source = Path(motion_source)
    return source.is_dir() or source.suffix in (".pkl", ".yaml", ".yml")

  def _resolve_motion_body_names(self) -> tuple[str, ...]:
    """Resolve body names for the motion tensors.

    Priority:
    1) Names embedded in the motion file (`body_names`/`body_link_names`).
    2) Names explicitly provided in the config (`motion_body_names`).
    3) Fallback to robot body names if tensor count matches exactly.
    """
    if self._uses_pkl_motion:
      assert self.motion_lib is not None
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
    if not self._uses_pkl_motion:
      return
    assert self.motion_lib is not None
    self._current_motion_frame = self.motion_lib.calc_motion_frame(
      self.motion_ids, self._current_times_s()
    )

  def query_motion_frames(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
    """Query future reference frames at given step offsets."""
    if len(step_offsets) == 0:
      raise ValueError("`step_offsets` must contain at least one entry.")

    if self._uses_pkl_motion:
      return self._query_motion_frames_pkl(step_offsets)
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

  def _query_motion_frames_pkl(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
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
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.joint_pos
    assert self.motion is not None
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.joint_vel
    assert self.motion is not None
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    if self._uses_pkl_motion:
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
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_quat_w[:, self.motion_body_indexes]
    assert self.motion is not None
    return self.motion.body_quat_w[self.time_steps][:, self.motion_body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_lin_vel_w[:, self.motion_body_indexes]
    assert self.motion is not None
    return self.motion.body_lin_vel_w[self.time_steps][:, self.motion_body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_ang_vel_w[:, self.motion_body_indexes]
    assert self.motion is not None
    return self.motion.body_ang_vel_w[self.time_steps][:, self.motion_body_indexes]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    if self._uses_pkl_motion:
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
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_quat_w[:, self.motion_anchor_body_index]
    assert self.motion is not None
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    if self._uses_pkl_motion:
      assert self._current_motion_frame is not None
      return self._current_motion_frame.body_lin_vel_w[:, self.motion_anchor_body_index]
    assert self.motion is not None
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    if self._uses_pkl_motion:
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
    if self._uses_pkl_motion:
      # Adaptive bins are defined over a single timeline; fallback to uniform for PKL sets.
      self._uniform_sampling(env_ids)
      return

    assert self.motion is not None
    episode_failed = self._env.termination_manager.terminated[env_ids]
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
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    if self._uses_pkl_motion:
      assert self.motion_lib is not None
      sampled_motion_ids = self.motion_lib.sample_motions(len(env_ids))
      self.motion_ids[env_ids] = sampled_motion_ids
      self.motion_time_offsets[env_ids] = self.motion_lib.sample_time(sampled_motion_ids)
      self.time_steps[env_ids] = 0
      self._refresh_motion_frame()
    else:
      assert self.motion is not None
      self.time_steps[env_ids] = torch.randint(
        0, self.motion.time_step_total, (len(env_ids),), device=self.device
      )

    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      if self._uses_pkl_motion:
        assert self.motion_lib is not None
        self.motion_ids[env_ids] = self.motion_lib.sample_motions(len(env_ids))
        self.motion_time_offsets[env_ids] = 0.0
      self.time_steps[env_ids] = 0
      if self._uses_pkl_motion:
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

    if self._uses_pkl_motion:
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

    if (not self._uses_pkl_motion) and self.cfg.sampling_mode == "adaptive":
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
