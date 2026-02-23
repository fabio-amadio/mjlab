"""Convert CLAMP legacy motion PKLs into mjlab-compatible NPZ motion files.

This utility is offline-only: it does not affect CLAMP runtime loading, which is
NPZ-only. It exists to regenerate NPZ datasets from historical PKL dumps.

Input PKL expected keys:
- ``root_pos``: (T, 3)
- ``root_rot``: (T, 4) root quaternion
- ``dof_pos``: (T, D)
- ``fps``: optional scalar, defaults to 30.0

Output NPZ keys (tracking/CLAMP-compatible):
- ``fps``: (1,)
- ``joint_pos``: (T, D)
- ``joint_vel``: (T, D)
- ``body_pos_w``: (T, B, 3)
- ``body_quat_w``: (T, B, 4) in ``wxyz``
- ``body_lin_vel_w``: (T, B, 3)
- ``body_ang_vel_w``: (T, B, 3)
- ``body_names``: (B,)
- ``body_link_names``: (B,)
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
from tqdm import tqdm

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML


QuatConvention = Literal["xyzw", "wxyz"]


@dataclass(frozen=True)
class Args:
  input_root: Path
  output_root: Path
  model: Path
  input_quat_convention: QuatConvention
  overwrite: bool
  compressed: bool
  fail_on_error: bool


def _default_input_root() -> Path:
  return Path(__file__).resolve().parents[5] / "assets" / "motions" / "clamp" / "g1_motions_pkls"


def _default_output_root() -> Path:
  return Path(__file__).resolve().parents[5] / "assets" / "motions" / "clamp" / "g1_motions_npz"


def _parse_args() -> Args:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-root",
    type=Path,
    default=_default_input_root(),
    help="Root folder containing PKLs (subfolders are preserved).",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=_default_output_root(),
    help="Root folder where converted NPZ files are written.",
  )
  parser.add_argument(
    "--model",
    type=Path,
    default=G1_XML,
    help="MuJoCo XML model used for FK (defaults to mjlab G1 model).",
  )
  parser.add_argument(
    "--input-quat-convention",
    choices=("xyzw", "wxyz"),
    default="wxyz",
    help="Quaternion convention used in PKL `root_rot`.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing NPZ files.",
  )
  parser.add_argument(
    "--uncompressed",
    action="store_true",
    help="Use np.savez (default uses np.savez_compressed).",
  )
  parser.add_argument(
    "--fail-on-error",
    action="store_true",
    help="Stop on first conversion error.",
  )
  ns = parser.parse_args()

  return Args(
    input_root=ns.input_root.expanduser().resolve(),
    output_root=ns.output_root.expanduser().resolve(),
    model=ns.model.expanduser().resolve(),
    input_quat_convention=ns.input_quat_convention,
    overwrite=ns.overwrite,
    compressed=not ns.uncompressed,
    fail_on_error=ns.fail_on_error,
  )


def _to_wxyz(quat: np.ndarray, convention: QuatConvention) -> np.ndarray:
  if convention == "wxyz":
    return quat
  if convention == "xyzw":
    return np.roll(quat, shift=1, axis=-1)
  raise ValueError(f"Unsupported quaternion convention: {convention}")


def _quat_normalize(q: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(q, axis=-1, keepdims=True)
  norm = np.clip(norm, 1.0e-12, None)
  return q / norm


def _quat_conj(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[..., 1:] *= -1.0
  return out


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    axis=-1,
  )


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
  q = _quat_normalize(q)
  w = np.clip(q[..., 0], -1.0, 1.0)
  v = q[..., 1:]
  v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
  small = v_norm < 1.0e-8

  angle = 2.0 * np.arctan2(v_norm[..., 0], w)
  axis = np.zeros_like(v)
  np.divide(v, v_norm, out=axis, where=~small)
  rotvec = axis * angle[..., None]

  # Small-angle fallback: sin(theta/2) ~= theta/2 => rotvec ~= 2*v
  rotvec = np.where(small, 2.0 * v, rotvec)
  return rotvec


def _finite_difference(x: np.ndarray, dt: float) -> np.ndarray:
  if x.shape[0] < 2:
    return np.zeros_like(x, dtype=np.float32)
  return np.gradient(x, dt, axis=0).astype(np.float32)


def _quat_angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  # quat_wxyz shape: (T, ..., 4)
  num_frames = quat_wxyz.shape[0]
  if num_frames < 2:
    return np.zeros((*quat_wxyz.shape[:-1], 3), dtype=np.float32)

  q_prev = quat_wxyz[:-2]
  q_next = quat_wxyz[2:].copy()

  # Hemisphere correction before differencing.
  flip = np.sum(q_next * q_prev, axis=-1, keepdims=True) < 0.0
  q_next = np.where(flip, -q_next, q_next)

  q_rel = _quat_mul(q_next, _quat_conj(q_prev))
  rotvec = _quat_to_rotvec(q_rel)
  omega = rotvec / (2.0 * dt)

  # Pad boundaries by replication to keep shape T.
  omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
  return omega.astype(np.float32)


def _joint_qpos_indices(model: mujoco.MjModel) -> tuple[int, list[int]]:
  free_joint_qadr: int | None = None
  joint_q_indices: list[int] = []
  for jid in range(model.njnt):
    jtype = model.jnt_type[jid]
    qadr = int(model.jnt_qposadr[jid])
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
      if free_joint_qadr is None:
        free_joint_qadr = qadr
      continue
    if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      joint_q_indices.append(qadr)
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
      joint_q_indices.extend(range(qadr, qadr + 4))
    else:
      raise ValueError(f"Unsupported joint type in FK model: {jtype}")

  if free_joint_qadr is None:
    raise ValueError("FK model has no free joint.")
  return free_joint_qadr, sorted(joint_q_indices)


def _robot_body_ids_and_names(model: mujoco.MjModel) -> tuple[list[int], tuple[str, ...]]:
  body_ids: list[int] = []
  body_names: list[str] = []
  for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    if name is None or name == "world":
      continue
    body_ids.append(bid)
    body_names.append(name)
  if len(body_ids) == 0:
    raise ValueError("No named non-world bodies found in model.")
  return body_ids, tuple(body_names)


def _load_pkl(path: Path) -> dict[str, object]:
  with open(path, "rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Invalid PKL object type in {path}: {type(data)}")
  return data


def _convert_one(
  pkl_path: Path,
  out_path: Path,
  args: Args,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  free_qadr: int,
  joint_q_indices: list[int],
  body_ids: list[int],
  body_names: tuple[str, ...],
) -> tuple[bool, str]:
  if out_path.exists() and not args.overwrite:
    return False, "skip: npz exists"

  motion = _load_pkl(pkl_path)
  required_keys = {"root_pos", "root_rot", "dof_pos"}
  missing = sorted(required_keys.difference(set(motion.keys())))
  if missing:
    return False, f"skip: missing keys {missing}"

  root_pos = np.asarray(motion["root_pos"], dtype=np.float64)
  root_rot = np.asarray(motion["root_rot"], dtype=np.float64)
  dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)

  if root_pos.ndim != 2 or root_pos.shape[1] != 3:
    raise ValueError(f"Invalid root_pos shape: {root_pos.shape}")
  if root_rot.ndim != 2 or root_rot.shape[1] != 4:
    raise ValueError(f"Invalid root_rot shape: {root_rot.shape}")
  if dof_pos.ndim != 2:
    raise ValueError(f"Invalid dof_pos shape: {dof_pos.shape}")
  if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
    raise ValueError(
      "Inconsistent frame count: "
      f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
    )
  if len(joint_q_indices) != dof_pos.shape[1]:
    raise ValueError(
      "DoF mismatch between model and PKL: "
      f"model={len(joint_q_indices)} pkl={dof_pos.shape[1]}"
    )

  fps_raw = motion.get("fps", 30.0)
  fps = float(np.asarray(fps_raw).reshape(-1)[0])
  if fps <= 1.0e-6:
    fps = 30.0
  dt = 1.0 / fps

  root_rot_wxyz = _quat_normalize(_to_wxyz(root_rot, args.input_quat_convention))
  num_frames = root_pos.shape[0]
  num_bodies = len(body_ids)

  body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
  qpos = np.zeros(model.nq, dtype=np.float64)
  joint_q_indices_arr = np.asarray(joint_q_indices, dtype=np.int64)

  for frame_id in range(num_frames):
    qpos[:] = 0.0
    qpos[free_qadr : free_qadr + 3] = root_pos[frame_id]
    qpos[free_qadr + 3 : free_qadr + 7] = root_rot_wxyz[frame_id]
    qpos[joint_q_indices_arr] = dof_pos[frame_id]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for i, bid in enumerate(body_ids):
      body_pos_w[frame_id, i] = data.xpos[bid]
      body_quat_w[frame_id, i] = data.xquat[bid]

  body_quat_w = _quat_normalize(body_quat_w).astype(np.float32)
  joint_pos = dof_pos.astype(np.float32)
  joint_vel = _finite_difference(joint_pos, dt)
  body_lin_vel_w = _finite_difference(body_pos_w, dt)
  body_ang_vel_w = _quat_angular_velocity(body_quat_w.astype(np.float64), dt)

  payload = {
    "fps": np.asarray([fps], dtype=np.float64),
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
    "body_names": np.asarray(body_names),
    "body_link_names": np.asarray(body_names),
  }

  out_path.parent.mkdir(parents=True, exist_ok=True)
  if args.compressed:
    np.savez_compressed(out_path, **payload)
  else:
    np.savez(out_path, **payload)
  return True, "ok"


def main() -> None:
  args = _parse_args()
  if not args.input_root.exists():
    raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
  if not args.model.exists():
    raise FileNotFoundError(f"Model XML does not exist: {args.model}")

  model = mujoco.MjModel.from_xml_path(str(args.model))
  data = mujoco.MjData(model)
  free_qadr, joint_q_indices = _joint_qpos_indices(model)
  body_ids, body_names = _robot_body_ids_and_names(model)

  pkl_files = sorted(args.input_root.rglob("*.pkl"))
  if len(pkl_files) == 0:
    print(f"No PKL files found in: {args.input_root}")
    return

  converted = 0
  skipped = 0
  failed = 0
  progress = tqdm(pkl_files, desc="Converting PKL->NPZ", unit="file")
  for pkl_path in progress:
    rel = pkl_path.relative_to(args.input_root)
    out_path = (args.output_root / rel).with_suffix(".npz")
    try:
      changed, msg = _convert_one(
        pkl_path=pkl_path,
        out_path=out_path,
        args=args,
        model=model,
        data=data,
        free_qadr=free_qadr,
        joint_q_indices=joint_q_indices,
        body_ids=body_ids,
        body_names=body_names,
      )
      if changed:
        converted += 1
      else:
        skipped += 1
      progress.set_postfix_str(f"converted={converted} skipped={skipped} failed={failed}")
      if msg != "ok":
        tqdm.write(f"{pkl_path}: {msg}")
    except Exception as exc:  # noqa: BLE001
      failed += 1
      tqdm.write(f"{pkl_path}: failed: {exc}")
      if args.fail_on_error:
        raise

  print(
    "Done. "
    f"converted={converted} skipped={skipped} failed={failed} total={len(pkl_files)}"
  )


if __name__ == "__main__":
  main()
