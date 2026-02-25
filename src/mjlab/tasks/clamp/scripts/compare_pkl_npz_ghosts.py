"""Overlay NPZ and PKL reference ghosts to inspect conversion differences.

This script runs CLAMP play mode with a zero policy and overrides the motion
command debug visualization to draw:
1) NPZ reference ghost (standard CLAMP ghost).
2) PKL reference ghost for the matched clip/time (second color).

Pairing rule:
- Resolve NPZ clips from --npz-motion-source (yaml/dir/file).
- Map each NPZ path to PKL via relative path:
  <npz_root>/<subdirs>/file.npz -> <pkl_root>/<subdirs>/file.pkl
"""

from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Literal, cast

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.clamp.mdp.motion_command import MotionCommand
from mjlab.tasks.clamp.mdp.motion_library import NpzMotionLibrary
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

_DEFAULT_NPZ_MOTION_SOURCE = str(
  Path(__file__).resolve().parents[1] / "config" / "g1" / "motion_data_cfg.yaml"
)
_DEFAULT_NPZ_ROOT = str(
  Path(__file__).resolve().parents[5] / "assets" / "motions" / "clamp" / "g1_motions_npz"
)
_DEFAULT_PKL_ROOT = str(
  Path(__file__).resolve().parents[5] / "assets" / "motions" / "clamp" / "g1_motions_pkl"
)


@dataclass(frozen=True)
class CompareGhostsConfig:
  task_id: str = "Mjlab-CLAMP-Teacher-Flat-Unitree-G1"
  device: str | None = None
  num_envs: int = 1
  viewer: Literal["auto", "native", "viser"] = "auto"
  npz_motion_source: str = _DEFAULT_NPZ_MOTION_SOURCE
  npz_root: str = _DEFAULT_NPZ_ROOT
  pkl_root: str = _DEFAULT_PKL_ROOT
  pkl_quat_convention: Literal["xyzw", "wxyz"] = "xyzw"
  npz_rgba: str = "0.2,0.7,1.0,0.45"
  pkl_rgba: str = "1.0,0.45,0.2,0.45"


@dataclass
class _PklClip:
  root_pos: np.ndarray  # [T,3], world
  root_rot_wxyz: np.ndarray  # [T,4], world
  dof_pos: np.ndarray  # [T,D]
  fps: float

  @property
  def num_frames(self) -> int:
    return int(self.root_pos.shape[0])

  @property
  def length_s(self) -> float:
    return float(max(self.num_frames - 1, 0)) / max(self.fps, 1.0e-6)


def _parse_rgba(value: str) -> np.ndarray:
  parts = [p.strip() for p in value.split(",")]
  if len(parts) != 4:
    raise ValueError(f"RGBA must have 4 comma-separated values, got: {value}")
  rgba = np.asarray([float(p) for p in parts], dtype=np.float32)
  return np.clip(rgba, 0.0, 1.0)


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
  n = np.linalg.norm(q, axis=-1, keepdims=True)
  return q / np.clip(n, 1.0e-12, None)


def _to_wxyz(q: np.ndarray, convention: Literal["xyzw", "wxyz"]) -> np.ndarray:
  if convention == "wxyz":
    return q
  return np.roll(q, shift=1, axis=-1)


def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, blend: float) -> np.ndarray:
  q0 = _normalize_quat_wxyz(q0)
  q1 = _normalize_quat_wxyz(q1)
  dot = float(np.dot(q0, q1))
  if dot < 0.0:
    q1 = -q1
    dot = -dot
  dot = float(np.clip(dot, 0.0, 1.0))

  if dot > 0.9995:
    out = (1.0 - blend) * q0 + blend * q1
    return _normalize_quat_wxyz(out)

  theta_0 = float(np.arccos(dot))
  sin_theta_0 = float(np.sin(theta_0))
  theta = theta_0 * blend
  s0 = np.sin(theta_0 - theta) / max(sin_theta_0, 1.0e-8)
  s1 = np.sin(theta) / max(sin_theta_0, 1.0e-8)
  out = s0 * q0 + s1 * q1
  return _normalize_quat_wxyz(out)


def _load_pkl_clip(
  pkl_path: Path, quat_convention: Literal["xyzw", "wxyz"]
) -> _PklClip:
  with open(pkl_path, "rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"PKL is not a dict: {pkl_path}")

  required = ("root_pos", "root_rot", "dof_pos")
  missing = [k for k in required if k not in data]
  if missing:
    raise ValueError(f"PKL missing keys {missing}: {pkl_path}")

  root_pos = np.asarray(data["root_pos"], dtype=np.float64)
  root_rot = np.asarray(data["root_rot"], dtype=np.float64)
  dof_pos = np.asarray(data["dof_pos"], dtype=np.float64)
  if root_pos.ndim != 2 or root_pos.shape[1] != 3:
    raise ValueError(f"Invalid root_pos shape in {pkl_path}: {root_pos.shape}")
  if root_rot.ndim != 2 or root_rot.shape[1] != 4:
    raise ValueError(f"Invalid root_rot shape in {pkl_path}: {root_rot.shape}")
  if dof_pos.ndim != 2:
    raise ValueError(f"Invalid dof_pos shape in {pkl_path}: {dof_pos.shape}")
  if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
    raise ValueError(f"Inconsistent frame count in {pkl_path}")

  fps = float(data.get("fps", 30.0))
  root_rot_wxyz = _normalize_quat_wxyz(_to_wxyz(root_rot, quat_convention))
  return _PklClip(root_pos=root_pos, root_rot_wxyz=root_rot_wxyz, dof_pos=dof_pos, fps=fps)


def _interp_clip(clip: _PklClip, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if clip.num_frames <= 1:
    return (
      clip.root_pos[0].copy(),
      clip.root_rot_wxyz[0].copy(),
      clip.dof_pos[0].copy(),
    )
  t = float(np.clip(t, 0.0, max(clip.length_s - 1.0e-6, 0.0)))
  phase = t / max(clip.length_s, 1.0e-6)
  idx0 = int(np.floor(phase * (clip.num_frames - 1)))
  idx1 = min(idx0 + 1, clip.num_frames - 1)
  blend = phase * (clip.num_frames - 1) - idx0

  root_pos = (1.0 - blend) * clip.root_pos[idx0] + blend * clip.root_pos[idx1]
  root_rot = _slerp_wxyz(clip.root_rot_wxyz[idx0], clip.root_rot_wxyz[idx1], float(blend))
  dof_pos = (1.0 - blend) * clip.dof_pos[idx0] + blend * clip.dof_pos[idx1]
  return root_pos, root_rot, dof_pos


def run(cfg: CompareGhostsConfig) -> None:
  # Ensure task registry is populated.
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  agent_cfg = load_rl_cfg(cfg.task_id)
  env_cfg.scene.num_envs = int(max(cfg.num_envs, 1))

  motion_cmd_cfg = env_cfg.commands.get("motion")
  if motion_cmd_cfg is None:
    raise ValueError(f"Task does not define a 'motion' command: {cfg.task_id}")
  if not hasattr(motion_cmd_cfg, "motion_file"):
    raise ValueError(
      f"Task motion command has no 'motion_file' field: {type(motion_cmd_cfg).__name__}"
    )
  motion_cmd_cfg.motion_file = cfg.npz_motion_source
  motion_cmd_cfg.debug_vis = True

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  command = cast(MotionCommand, env.unwrapped.command_manager.get_term("motion"))
  npz_files, _ = NpzMotionLibrary._resolve_motion_entries(cfg.npz_motion_source)
  if command.motion_lib is not None:
    if len(npz_files) != command.motion_lib.num_motions():
      raise RuntimeError(
        "Resolved NPZ file count does not match loaded motion_lib num_motions."
      )
  elif len(npz_files) != 1:
    raise ValueError(
      "Single-file motion command is active, but multiple NPZ files were resolved. "
      "Use one .npz with --npz-motion-source or use a YAML/dir source."
    )

  npz_root = Path(cfg.npz_root).expanduser().resolve()
  pkl_root = Path(cfg.pkl_root).expanduser().resolve()

  pkl_paths: list[Path | None] = []
  missing_pairs = 0
  for npz_path in npz_files:
    npz_path = npz_path.resolve()
    try:
      rel = npz_path.relative_to(npz_root)
      pkl_path = (pkl_root / rel).with_suffix(".pkl")
    except ValueError:
      # Fallback for single-file mode: match by stem under pkl_root.
      if len(npz_files) == 1:
        matches = sorted(pkl_root.rglob(f"{npz_path.stem}.pkl"))
        pkl_path = matches[0] if len(matches) == 1 else None
      else:
        pkl_path = None
    if pkl_path is None or not pkl_path.exists():
      pkl_paths.append(None)
      missing_pairs += 1
    else:
      pkl_paths.append(pkl_path)

  print(f"[INFO] Device: {device}")
  print(f"[INFO] NPZ motions loaded: {len(npz_files)}")
  print(f"[INFO] PKL matches found: {len(npz_files) - missing_pairs}")
  print(f"[INFO] PKL missing matches: {missing_pairs}")
  if missing_pairs > 0:
    print(
      "[WARN] Missing PKL pairs will skip PKL ghost for those motion ids. "
      "Check --npz-root/--pkl-root mapping."
    )

  npz_rgba = _parse_rgba(cfg.npz_rgba)
  pkl_rgba = _parse_rgba(cfg.pkl_rgba)
  command._ghost_color = npz_rgba
  command._ghost_model = None
  pkl_ghost_model = copy.deepcopy(env.unwrapped.sim.mj_model)
  pkl_ghost_model.geom_rgba[:] = pkl_rgba

  robot_entity = env.unwrapped.scene[command.cfg.entity_name]
  free_joint_q_adr = robot_entity.indexing.free_joint_q_adr.cpu().numpy()
  joint_q_adr = robot_entity.indexing.joint_q_adr.cpu().numpy()
  expected_dof = int(joint_q_adr.shape[0])

  pkl_cache: dict[int, _PklClip] = {}
  warned_missing: set[int] = set()
  warned_bad_dof: set[int] = set()
  original_debug_impl = command._debug_vis_impl

  def _load_clip_for_motion_id(motion_id: int) -> _PklClip | None:
    if motion_id in pkl_cache:
      return pkl_cache[motion_id]
    pkl_path = pkl_paths[motion_id]
    if pkl_path is None:
      return None
    clip = _load_pkl_clip(pkl_path, cfg.pkl_quat_convention)
    pkl_cache[motion_id] = clip
    return clip

  def _debug_vis_compare(self: MotionCommand, visualizer) -> None:
    original_debug_impl(visualizer)
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    step_dt = float(self._env.step_dt)
    for env_id in env_indices:
      motion_id = int(self.motion_ids[env_id].item())
      clip = _load_clip_for_motion_id(motion_id)
      if clip is None:
        if motion_id not in warned_missing:
          warned_missing.add(motion_id)
          print(f"[WARN] No PKL pair for motion_id={motion_id}; skipping PKL ghost.")
        continue

      if clip.dof_pos.shape[1] != expected_dof:
        if motion_id not in warned_bad_dof:
          warned_bad_dof.add(motion_id)
          print(
            f"[WARN] DoF mismatch for motion_id={motion_id}: "
            f"pkl={clip.dof_pos.shape[1]} expected={expected_dof}; skipping PKL ghost."
          )
        continue

      t = float(self.time_steps[env_id].item()) * step_dt + float(
        self.motion_time_offsets[env_id].item()
      )
      root_pos, root_rot, dof_pos = _interp_clip(clip, t)

      qpos = np.zeros(self._env.sim.mj_model.nq, dtype=np.float64)
      qpos[free_joint_q_adr[0:3]] = root_pos + self._env.scene.env_origins[env_id].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = root_rot
      qpos[joint_q_adr] = dof_pos
      visualizer.add_ghost_mesh(qpos, model=pkl_ghost_model, label=f"pkl_ghost_{env_id}")

  command._debug_vis_impl = MethodType(_debug_vis_compare, command)

  action_shape = env.unwrapped.action_space.shape  # type: ignore[attr-defined]

  class _ZeroPolicy:
    def __call__(self, obs) -> torch.Tensor:
      del obs
      return torch.zeros(action_shape, device=env.unwrapped.device)

  policy = _ZeroPolicy()

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    viewer = "native" if has_display else "viser"
  else:
    viewer = cfg.viewer

  print(
    "[INFO] NPZ ghost color RGBA="
    f"{tuple(float(x) for x in npz_rgba)} | "
    "PKL ghost color RGBA="
    f"{tuple(float(x) for x in pkl_rgba)}"
  )
  if viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise ValueError(f"Unsupported viewer: {viewer}")

  env.close()


def main() -> None:
  cfg = tyro.cli(
    CompareGhostsConfig,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run(cfg)


if __name__ == "__main__":
  main()
