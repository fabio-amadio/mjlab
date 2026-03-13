"""Motion-data loading and library query utilities for CLAMP."""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .indexing import build_name_to_index

try:
  from tqdm import tqdm
except ModuleNotFoundError:
  tqdm = None


class MotionLoader:
  """Load a single motion NPZ file into torch tensors."""

  def __init__(
    self,
    motion_file: str,
    device: str = "cpu",
    required_body_names: tuple[str, ...] | None = None,
  ) -> None:
    with np.load(motion_file) as data:
      file_body_names = self._extract_body_names(data)
      selected_indices: np.ndarray | None = None
      if required_body_names is not None:
        if file_body_names is None:
          raise ValueError(
            "Motion npz must include body names (`body_names` or `body_link_names`) "
            f"when selective loading is enabled: {motion_file}"
          )
        selected_indices = _resolve_required_body_indices(
          file_body_names=file_body_names,
          required_body_names=required_body_names,
          source=motion_file,
        )

      self.joint_pos = torch.tensor(
        data["joint_pos"], dtype=torch.float32, device=device
      )
      self.joint_vel = torch.tensor(
        data["joint_vel"], dtype=torch.float32, device=device
      )
      body_pos_w = np.asarray(data["body_pos_w"])
      body_quat_w = np.asarray(data["body_quat_w"])
      body_lin_vel_w = np.asarray(data["body_lin_vel_w"])
      body_ang_vel_w = np.asarray(data["body_ang_vel_w"])
      if selected_indices is not None:
        body_pos_w = body_pos_w[:, selected_indices, :]
        body_quat_w = body_quat_w[:, selected_indices, :]
        body_lin_vel_w = body_lin_vel_w[:, selected_indices, :]
        body_ang_vel_w = body_ang_vel_w[:, selected_indices, :]
      self.body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
      self.body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
      self.body_lin_vel_w = torch.tensor(
        body_lin_vel_w, dtype=torch.float32, device=device
      )
      self.body_ang_vel_w = torch.tensor(
        body_ang_vel_w, dtype=torch.float32, device=device
      )
      if required_body_names is None:
        self.body_names = file_body_names
      else:
        self.body_names = tuple(required_body_names)

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
  """Batch of queried motion-reference tensors in [env, ...] form."""

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
    _env_int("RANK", 0) == 0 and _env_int("LOCAL_RANK", 0) == 0 and sys.stderr.isatty()
  )


def _load_yaml_config(yaml_path: Path) -> dict[str, object]:
  import yaml

  with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"YAML config must be a mapping at top level: {yaml_path}")
  return data


def _resolve_required_body_indices(
  file_body_names: tuple[str, ...],
  required_body_names: tuple[str, ...],
  source: str | Path,
) -> np.ndarray:
  if len(required_body_names) == 0:
    raise ValueError("`required_body_names` must be non-empty when provided.")
  unique_required = tuple(dict.fromkeys(required_body_names))
  if len(unique_required) != len(required_body_names):
    raise ValueError(
      f"`required_body_names` contains duplicates for `{source}`: {required_body_names}"
    )

  name_to_index = build_name_to_index(file_body_names, str(source))
  missing = [name for name in required_body_names if name not in name_to_index]
  if missing:
    raise ValueError(
      f"Missing required motion bodies in `{source}`: {missing}. "
      f"Available bodies: {file_body_names}"
    )
  return np.asarray(
    [name_to_index[name] for name in required_body_names], dtype=np.int64
  )


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
    required_body_names: tuple[str, ...] | None = None,
  ) -> None:
    self.device = device
    self.required_body_names = required_body_names
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
            f"Invalid motion npz. Missing keys: {sorted(missing_keys)} in {motion_file}"
          )

        fps = self._extract_fps(data)
        dt = 1.0 / max(fps, 1e-6)
        file_body_names = MotionLoader._extract_body_names(data)
        if file_body_names is None:
          raise ValueError(
            "Motion npz must include body names (`body_names` or `body_link_names`): "
            f"{motion_file}"
          )
        selected_indices: np.ndarray | None = None
        selected_body_names: tuple[str, ...]
        if self.required_body_names is not None:
          selected_indices = _resolve_required_body_indices(
            file_body_names=file_body_names,
            required_body_names=self.required_body_names,
            source=motion_file,
          )
          selected_body_names = tuple(self.required_body_names)
        else:
          selected_body_names = tuple(file_body_names)

        joint_pos = torch.tensor(
          np.asarray(data["joint_pos"]), dtype=torch.float32, device=self.device
        )
        joint_vel = torch.tensor(
          np.asarray(data["joint_vel"]), dtype=torch.float32, device=self.device
        )
        body_pos_w_np = np.asarray(data["body_pos_w"])
        body_quat_w_np = np.asarray(data["body_quat_w"])
        body_lin_vel_w_np = np.asarray(data["body_lin_vel_w"])
        body_ang_vel_w_np = np.asarray(data["body_ang_vel_w"])
        if selected_indices is not None:
          body_pos_w_np = body_pos_w_np[:, selected_indices, :]
          body_quat_w_np = body_quat_w_np[:, selected_indices, :]
          body_lin_vel_w_np = body_lin_vel_w_np[:, selected_indices, :]
          body_ang_vel_w_np = body_ang_vel_w_np[:, selected_indices, :]
        body_pos_w = torch.tensor(
          body_pos_w_np, dtype=torch.float32, device=self.device
        )
        body_quat_w = torch.tensor(
          body_quat_w_np, dtype=torch.float32, device=self.device
        )
        body_lin_vel_w = torch.tensor(
          body_lin_vel_w_np, dtype=torch.float32, device=self.device
        )
        body_ang_vel_w = torch.tensor(
          body_ang_vel_w_np, dtype=torch.float32, device=self.device
        )

      if joint_pos.ndim != 2:
        raise ValueError(f"Invalid joint_pos shape in {motion_file}: {joint_pos.shape}")
      if joint_vel.shape != joint_pos.shape:
        raise ValueError(
          f"joint_vel must match joint_pos shape in {motion_file}: "
          f"{joint_vel.shape} vs {joint_pos.shape}"
        )
      if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(
          f"Invalid body_pos_w shape in {motion_file}: {body_pos_w.shape}"
        )
      if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(
          f"Invalid body_quat_w shape in {motion_file}: {body_quat_w.shape}"
        )
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

      if self.body_names is None:
        self.body_names = selected_body_names
      elif self.body_names != selected_body_names:
        raise ValueError(
          "All NPZ files must share the same selected body name ordering. "
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
    entries = list(zip(motion_files, per_motion_weights, strict=True))
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
    """Return the number of clips loaded into the library."""
    return int(self.motion_num_frames.shape[0])

  def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Return clip lengths in seconds for the given motion ids."""
    return self.motion_lengths_s[motion_ids]

  def sample_motions(self, n: int) -> torch.Tensor:
    """Sample motion ids according to per-clip weights."""
    return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

  def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Sample a random time uniformly within each selected clip."""
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
    self,
    motion_ids: torch.Tensor,
    motion_times: torch.Tensor,
    anchor_body_index: int,
  ) -> MotionFrameBatch:
    """Interpolate and return motion-reference tensors at the requested times."""
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
    if not (0 <= anchor_body_index < num_bodies):
      raise ValueError(
        f"Invalid anchor_body_index={anchor_body_index}. "
        f"Expected in [0, {num_bodies - 1}]."
      )

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
      anchor_pos_w=body_pos_w[:, anchor_body_index],
      anchor_quat_w=body_quat_w[:, anchor_body_index],
      anchor_lin_vel_w=body_lin_vel[:, anchor_body_index],
      anchor_ang_vel_w=body_ang_vel[:, anchor_body_index],
    )
