"""Verify PKL -> NPZ motion dataset porting quality.

This script compares legacy PKL files against converted NPZ files by matching
relative paths (same subfolder/file name, different extension).

It is robust to expected dataset differences:
- PKL may have different DoF count than NPZ.
- PKL may include more body names than NPZ.

For each matched pair, it reports:
- root position error (PKL root vs NPZ root body)
- root orientation error in degrees
- body position error for common body names
- joint position error (only when DoF dimensions match)
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

QuatConvention = Literal["xyzw", "wxyz", "auto"]


@dataclass(frozen=True)
class Args:
  pkl_root: Path
  npz_root: Path
  root_body_name: str
  pkl_quat_convention: QuatConvention
  max_files: int | None
  report_json: Path | None
  top_k: int
  fail_root_pos_max: float | None
  fail_root_ori_deg_max: float | None
  fail_body_pos_max: float | None
  fail_joint_pos_max: float | None


@dataclass
class FileMetrics:
  rel_path: str
  status: str
  message: str = ""
  quat_convention_used: str = ""
  num_frames_used: int = 0
  pkl_frames: int = 0
  npz_frames: int = 0
  pkl_dofs: int = 0
  npz_dofs: int = 0
  pkl_bodies: int = 0
  npz_bodies: int = 0
  common_bodies: int = 0
  missing_bodies_in_npz: int = 0
  extra_bodies_in_npz: int = 0
  root_pos_mae: float | None = None
  root_pos_max: float | None = None
  root_ori_deg_mae: float | None = None
  root_ori_deg_max: float | None = None
  body_pos_mae: float | None = None
  body_pos_max: float | None = None
  joint_pos_mae: float | None = None
  joint_pos_max: float | None = None


def _default_pkl_root() -> Path:
  return (
    Path(__file__).resolve().parents[5] / "assets" / "motions" / "twist_motion_dataset"
  )


def _default_npz_root() -> Path:
  return (
    Path(__file__).resolve().parents[5]
    / "assets"
    / "motions"
    / "clamp"
    / "g1_motions_npz"
  )


def _parse_args() -> Args:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--pkl-root",
    type=Path,
    default=_default_pkl_root(),
    help="Root folder containing source PKL files.",
  )
  parser.add_argument(
    "--npz-root",
    type=Path,
    default=_default_npz_root(),
    help="Root folder containing converted NPZ files.",
  )
  parser.add_argument(
    "--root-body-name",
    type=str,
    default="pelvis",
    help="Body name to use as root reference in NPZ.",
  )
  parser.add_argument(
    "--pkl-quat-convention",
    choices=("xyzw", "wxyz", "auto"),
    default="auto",
    help="Convention for PKL root_rot. 'auto' selects convention with lower root-orientation error.",
  )
  parser.add_argument(
    "--max-files",
    type=int,
    default=None,
    help="Optional cap on number of PKL files to process.",
  )
  parser.add_argument(
    "--report-json",
    type=Path,
    default=None,
    help="Optional path to save full JSON report.",
  )
  parser.add_argument(
    "--top-k",
    type=int,
    default=10,
    help="Number of worst files to print per metric.",
  )
  parser.add_argument(
    "--fail-root-pos-max",
    type=float,
    default=None,
    help="Fail (exit code 1) if any file exceeds this root position max error.",
  )
  parser.add_argument(
    "--fail-root-ori-deg-max",
    type=float,
    default=None,
    help="Fail (exit code 1) if any file exceeds this root orientation max error (deg).",
  )
  parser.add_argument(
    "--fail-body-pos-max",
    type=float,
    default=None,
    help="Fail (exit code 1) if any file exceeds this body position max error.",
  )
  parser.add_argument(
    "--fail-joint-pos-max",
    type=float,
    default=None,
    help="Fail (exit code 1) if any file exceeds this joint position max error.",
  )
  ns = parser.parse_args()

  return Args(
    pkl_root=ns.pkl_root.expanduser().resolve(),
    npz_root=ns.npz_root.expanduser().resolve(),
    root_body_name=ns.root_body_name,
    pkl_quat_convention=ns.pkl_quat_convention,
    max_files=ns.max_files,
    report_json=ns.report_json.expanduser().resolve()
    if ns.report_json is not None
    else None,
    top_k=max(1, int(ns.top_k)),
    fail_root_pos_max=ns.fail_root_pos_max,
    fail_root_ori_deg_max=ns.fail_root_ori_deg_max,
    fail_body_pos_max=ns.fail_body_pos_max,
    fail_joint_pos_max=ns.fail_joint_pos_max,
  )


def _to_wxyz(quat: np.ndarray, convention: Literal["xyzw", "wxyz"]) -> np.ndarray:
  if convention == "wxyz":
    return quat
  return np.roll(quat, shift=1, axis=-1)


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


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  qv = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=v.dtype), v], axis=-1)
  return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[..., 1:]


def _quat_error_deg(q_ref: np.ndarray, q_est: np.ndarray) -> np.ndarray:
  q_ref = _quat_normalize(q_ref)
  q_est = _quat_normalize(q_est)
  dot = np.sum(q_ref * q_est, axis=-1)
  dot = np.clip(np.abs(dot), 0.0, 1.0)
  return np.rad2deg(2.0 * np.arccos(dot))


def _load_pkl(path: Path) -> dict[str, object]:
  with open(path, "rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"PKL is not a dict: {path}")
  return data


def _decode_names(values: np.ndarray) -> list[str]:
  out: list[str] = []
  for value in values.reshape(-1).tolist():
    if isinstance(value, bytes):
      out.append(value.decode("utf-8"))
    else:
      out.append(str(value))
  return out


def _select_pkl_quat_wxyz(
  pkl_root_rot: np.ndarray,
  npz_root_quat_wxyz: np.ndarray,
  mode: QuatConvention,
) -> tuple[np.ndarray, str]:
  if mode in ("xyzw", "wxyz"):
    selected = _to_wxyz(pkl_root_rot, mode)
    return _quat_normalize(selected), mode

  q_xyzw = _quat_normalize(_to_wxyz(pkl_root_rot, "xyzw"))
  q_wxyz = _quat_normalize(_to_wxyz(pkl_root_rot, "wxyz"))
  err_xy = float(np.mean(_quat_error_deg(q_xyzw, npz_root_quat_wxyz)))
  err_w = float(np.mean(_quat_error_deg(q_wxyz, npz_root_quat_wxyz)))
  if err_xy <= err_w:
    return q_xyzw, "xyzw"
  return q_wxyz, "wxyz"


def _pair_metrics(args: Args, pkl_path: Path, npz_path: Path) -> FileMetrics:
  rel = str(pkl_path.relative_to(args.pkl_root))
  metrics = FileMetrics(rel_path=rel, status="ok")

  pkl = _load_pkl(pkl_path)
  with np.load(npz_path, allow_pickle=False) as npz:
    required_pkl = {
      "root_pos",
      "root_rot",
      "local_body_pos",
      "link_body_list",
      "dof_pos",
    }
    required_npz = {"joint_pos", "body_pos_w", "body_quat_w", "body_names"}
    missing_pkl = sorted(required_pkl.difference(set(pkl.keys())))
    missing_npz = sorted(required_npz.difference(set(npz.files)))
    if missing_pkl or missing_npz:
      msg = []
      if missing_pkl:
        msg.append(f"missing PKL keys: {missing_pkl}")
      if missing_npz:
        msg.append(f"missing NPZ keys: {missing_npz}")
      metrics.status = "error"
      metrics.message = "; ".join(msg)
      return metrics

    root_pos_pkl = np.asarray(pkl["root_pos"], dtype=np.float64)
    root_rot_pkl_raw = np.asarray(pkl["root_rot"], dtype=np.float64)
    dof_pos_pkl = np.asarray(pkl["dof_pos"], dtype=np.float64)
    local_body_pos_pkl = np.asarray(pkl["local_body_pos"], dtype=np.float64)
    body_names_pkl = [str(n) for n in pkl["link_body_list"]]

    joint_pos_npz = np.asarray(npz["joint_pos"], dtype=np.float64)
    body_pos_npz = np.asarray(npz["body_pos_w"], dtype=np.float64)
    body_quat_npz = _quat_normalize(np.asarray(npz["body_quat_w"], dtype=np.float64))
    body_names_npz = _decode_names(np.asarray(npz["body_names"]))

  metrics.pkl_frames = int(root_pos_pkl.shape[0])
  metrics.npz_frames = int(joint_pos_npz.shape[0])
  metrics.pkl_dofs = int(dof_pos_pkl.shape[1]) if dof_pos_pkl.ndim == 2 else 0
  metrics.npz_dofs = int(joint_pos_npz.shape[1]) if joint_pos_npz.ndim == 2 else 0
  metrics.pkl_bodies = len(body_names_pkl)
  metrics.npz_bodies = len(body_names_npz)

  if root_pos_pkl.ndim != 2 or root_pos_pkl.shape[1] != 3:
    metrics.status = "error"
    metrics.message = f"invalid root_pos shape {root_pos_pkl.shape}"
    return metrics
  if root_rot_pkl_raw.ndim != 2 or root_rot_pkl_raw.shape[1] != 4:
    metrics.status = "error"
    metrics.message = f"invalid root_rot shape {root_rot_pkl_raw.shape}"
    return metrics
  if local_body_pos_pkl.ndim != 3 or local_body_pos_pkl.shape[2] != 3:
    metrics.status = "error"
    metrics.message = f"invalid local_body_pos shape {local_body_pos_pkl.shape}"
    return metrics
  if body_pos_npz.ndim != 3 or body_pos_npz.shape[2] != 3:
    metrics.status = "error"
    metrics.message = f"invalid body_pos_w shape {body_pos_npz.shape}"
    return metrics
  if body_quat_npz.ndim != 3 or body_quat_npz.shape[2] != 4:
    metrics.status = "error"
    metrics.message = f"invalid body_quat_w shape {body_quat_npz.shape}"
    return metrics

  num_frames = min(
    root_pos_pkl.shape[0],
    root_rot_pkl_raw.shape[0],
    local_body_pos_pkl.shape[0],
    joint_pos_npz.shape[0],
    body_pos_npz.shape[0],
    body_quat_npz.shape[0],
  )
  metrics.num_frames_used = int(num_frames)
  if num_frames < 1:
    metrics.status = "error"
    metrics.message = "no frames available"
    return metrics

  if args.root_body_name in body_names_npz:
    root_idx_npz = body_names_npz.index(args.root_body_name)
  elif len(body_names_npz) > 0:
    root_idx_npz = 0
    metrics.message = f"root body `{args.root_body_name}` missing in NPZ; fallback to `{body_names_npz[0]}`"
  else:
    metrics.status = "error"
    metrics.message = "NPZ body_names is empty"
    return metrics

  root_pos_npz = body_pos_npz[:num_frames, root_idx_npz]
  root_quat_npz = body_quat_npz[:num_frames, root_idx_npz]
  root_rot_pkl_wxyz, quat_used = _select_pkl_quat_wxyz(
    root_rot_pkl_raw[:num_frames], root_quat_npz, args.pkl_quat_convention
  )
  metrics.quat_convention_used = quat_used

  root_pos_err = np.linalg.norm(root_pos_pkl[:num_frames] - root_pos_npz, axis=-1)
  root_ori_deg_err = _quat_error_deg(root_rot_pkl_wxyz, root_quat_npz)
  metrics.root_pos_mae = float(np.mean(root_pos_err))
  metrics.root_pos_max = float(np.max(root_pos_err))
  metrics.root_ori_deg_mae = float(np.mean(root_ori_deg_err))
  metrics.root_ori_deg_max = float(np.max(root_ori_deg_err))

  npz_index = {name: idx for idx, name in enumerate(body_names_npz)}
  pkl_index = {name: idx for idx, name in enumerate(body_names_pkl)}
  common_names = [name for name in body_names_npz if name in pkl_index]
  metrics.common_bodies = len(common_names)
  metrics.missing_bodies_in_npz = sum(name not in npz_index for name in body_names_pkl)
  metrics.extra_bodies_in_npz = sum(name not in pkl_index for name in body_names_npz)

  if len(common_names) > 0:
    pkl_ids = np.asarray([pkl_index[name] for name in common_names], dtype=np.int64)
    npz_ids = np.asarray([npz_index[name] for name in common_names], dtype=np.int64)
    world_body_pos_from_pkl = root_pos_pkl[:num_frames, None, :] + _quat_apply(
      root_rot_pkl_wxyz[:, None, :], local_body_pos_pkl[:num_frames, pkl_ids, :]
    )
    body_pos_err = np.linalg.norm(
      world_body_pos_from_pkl - body_pos_npz[:num_frames, npz_ids, :], axis=-1
    )
    metrics.body_pos_mae = float(np.mean(body_pos_err))
    metrics.body_pos_max = float(np.max(body_pos_err))

  if (
    dof_pos_pkl.ndim == 2
    and joint_pos_npz.ndim == 2
    and dof_pos_pkl.shape[1] == joint_pos_npz.shape[1]
  ):
    joint_pos_err = np.abs(dof_pos_pkl[:num_frames] - joint_pos_npz[:num_frames])
    metrics.joint_pos_mae = float(np.mean(joint_pos_err))
    metrics.joint_pos_max = float(np.max(joint_pos_err))

  return metrics


def _print_summary(results: list[FileMetrics], top_k: int) -> None:
  total = len(results)
  ok = [r for r in results if r.status == "ok"]
  missing = [r for r in results if r.status == "missing_npz"]
  failed = [r for r in results if r.status == "error"]

  print(f"Total PKL files processed: {total}")
  print(f"Matched and compared: {len(ok)}")
  print(f"Missing NPZ pair: {len(missing)}")
  print(f"Failed comparisons: {len(failed)}")

  def _agg(name: str):
    vals = [getattr(r, name) for r in ok if getattr(r, name) is not None]
    if len(vals) == 0:
      return None, None
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.mean(arr)), float(np.max(arr))

  for metric_name in (
    "root_pos_max",
    "root_ori_deg_max",
    "body_pos_max",
    "joint_pos_max",
  ):
    mean_val, max_val = _agg(metric_name)
    if mean_val is not None:
      print(f"{metric_name}: mean={mean_val:.6g} max={max_val:.6g}")

  def _top(metric_name: str):
    rows = [r for r in ok if getattr(r, metric_name) is not None]
    rows.sort(key=lambda r: float(getattr(r, metric_name)), reverse=True)
    return rows[:top_k]

  for metric_name in (
    "root_pos_max",
    "root_ori_deg_max",
    "body_pos_max",
    "joint_pos_max",
  ):
    rows = _top(metric_name)
    if len(rows) == 0:
      continue
    print(f"\nTop {len(rows)} by {metric_name}:")
    for row in rows:
      value = float(getattr(row, metric_name))
      print(f"  {value:.6g}  {row.rel_path}")

  if len(missing) > 0:
    print(f"\nFirst {min(top_k, len(missing))} missing NPZ pairs:")
    for row in missing[:top_k]:
      print(f"  {row.rel_path}")

  if len(failed) > 0:
    print(f"\nFirst {min(top_k, len(failed))} errors:")
    for row in failed[:top_k]:
      print(f"  {row.rel_path}: {row.message}")


def _check_thresholds(args: Args, results: list[FileMetrics]) -> list[str]:
  violations: list[str] = []
  ok = [r for r in results if r.status == "ok"]

  def _max_metric(name: str) -> tuple[float, str] | None:
    rows = [r for r in ok if getattr(r, name) is not None]
    if len(rows) == 0:
      return None
    row = max(rows, key=lambda r: float(getattr(r, name)))
    return float(getattr(row, name)), row.rel_path

  checks = [
    ("root_pos_max", args.fail_root_pos_max),
    ("root_ori_deg_max", args.fail_root_ori_deg_max),
    ("body_pos_max", args.fail_body_pos_max),
    ("joint_pos_max", args.fail_joint_pos_max),
  ]
  for metric_name, threshold in checks:
    if threshold is None:
      continue
    max_entry = _max_metric(metric_name)
    if max_entry is None:
      continue
    max_value, rel_path = max_entry
    if max_value > threshold:
      violations.append(
        f"{metric_name}={max_value:.6g} exceeds threshold={threshold:.6g} at {rel_path}"
      )
  return violations


def main() -> None:
  args = _parse_args()
  if not args.pkl_root.exists():
    raise FileNotFoundError(f"PKL root does not exist: {args.pkl_root}")
  if not args.npz_root.exists():
    raise FileNotFoundError(f"NPZ root does not exist: {args.npz_root}")

  pkl_files = sorted(args.pkl_root.rglob("*.pkl"))
  if args.max_files is not None:
    pkl_files = pkl_files[: max(0, args.max_files)]

  results: list[FileMetrics] = []
  for pkl_path in tqdm(pkl_files, desc="Verifying PKL->NPZ pairs", unit="file"):
    rel = pkl_path.relative_to(args.pkl_root)
    npz_path = (args.npz_root / rel).with_suffix(".npz")
    if not npz_path.exists():
      results.append(
        FileMetrics(
          rel_path=str(rel),
          status="missing_npz",
          message=f"missing pair: {npz_path}",
        )
      )
      continue

    try:
      results.append(_pair_metrics(args, pkl_path, npz_path))
    except Exception as exc:  # noqa: BLE001
      results.append(
        FileMetrics(
          rel_path=str(rel), status="error", message=f"{type(exc).__name__}: {exc}"
        )
      )

  _print_summary(results, args.top_k)

  if args.report_json is not None:
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
      json.dump(
        {
          "args": {
            "pkl_root": str(args.pkl_root),
            "npz_root": str(args.npz_root),
            "root_body_name": args.root_body_name,
            "pkl_quat_convention": args.pkl_quat_convention,
            "max_files": args.max_files,
            "top_k": args.top_k,
            "fail_root_pos_max": args.fail_root_pos_max,
            "fail_root_ori_deg_max": args.fail_root_ori_deg_max,
            "fail_body_pos_max": args.fail_body_pos_max,
            "fail_joint_pos_max": args.fail_joint_pos_max,
          },
          "results": [asdict(r) for r in results],
        },
        f,
      )
    print(f"\nWrote JSON report: {args.report_json}")

  violations = _check_thresholds(args, results)
  if len(violations) > 0:
    print("\nThreshold violations:")
    for violation in violations:
      print(f"  - {violation}")
    raise SystemExit(1)


if __name__ == "__main__":
  main()
