"""Prune NPZ motions where the robot becomes horizontal at any point.

Definition used:
- Let `u_z` be the world-frame z-component of the root body local +Z axis.
- A frame is considered "horizontal" when `abs(u_z) <= horizontal_abs_up_z_threshold`.
- A motion is flagged if this condition holds for at least
  `min_consecutive_frames` consecutive frames.

By default this script runs in dry-run mode and only reports candidates.
Use `--delete` to actually remove files, or `--trash-dir` to move them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class Args:
  npz_root: Path
  root_body_name: str
  horizontal_abs_up_z_threshold: float
  min_consecutive_frames: int
  delete: bool
  trash_dir: Path | None
  max_files: int | None
  report_json: Path | None


@dataclass
class FileResult:
  rel_path: str
  status: str
  message: str = ""
  num_frames: int = 0
  first_horizontal_frame: int | None = None
  longest_horizontal_run: int = 0
  min_abs_up_z: float | None = None
  action: str = "none"


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
    "--npz-root",
    type=Path,
    default=_default_npz_root(),
    help="Root folder containing NPZ motion files.",
  )
  parser.add_argument(
    "--root-body-name",
    type=str,
    default="pelvis",
    help="Body used to evaluate orientation (must exist in each NPZ body_names).",
  )
  parser.add_argument(
    "--horizontal-abs-up-z-threshold",
    type=float,
    default=0.30,
    help="Frame is horizontal if abs(up_z) <= this threshold.",
  )
  parser.add_argument(
    "--min-consecutive-frames",
    type=int,
    default=1,
    help="Minimum consecutive horizontal frames needed to flag a motion.",
  )
  parser.add_argument(
    "--delete",
    action="store_true",
    help="Actually delete/move flagged files. Default is dry-run.",
  )
  parser.add_argument(
    "--trash-dir",
    type=Path,
    default=None,
    help="If set with --delete, move flagged files here instead of unlinking.",
  )
  parser.add_argument(
    "--max-files",
    type=int,
    default=None,
    help="Optional cap on processed files (debug/smoke-test).",
  )
  parser.add_argument(
    "--report-json",
    type=Path,
    default=None,
    help="Optional output JSON path for full report.",
  )
  ns = parser.parse_args()

  min_consecutive_frames = max(1, int(ns.min_consecutive_frames))
  threshold = float(ns.horizontal_abs_up_z_threshold)
  if not (0.0 <= threshold <= 1.0):
    raise ValueError("--horizontal-abs-up-z-threshold must be in [0, 1].")
  trash_dir = ns.trash_dir.expanduser().resolve() if ns.trash_dir is not None else None
  report_json = (
    ns.report_json.expanduser().resolve() if ns.report_json is not None else None
  )
  return Args(
    npz_root=ns.npz_root.expanduser().resolve(),
    root_body_name=ns.root_body_name,
    horizontal_abs_up_z_threshold=threshold,
    min_consecutive_frames=min_consecutive_frames,
    delete=bool(ns.delete),
    trash_dir=trash_dir,
    max_files=ns.max_files,
    report_json=report_json,
  )


def _decode_names(values: np.ndarray) -> list[str]:
  names: list[str] = []
  for value in values.reshape(-1).tolist():
    if isinstance(value, bytes):
      names.append(value.decode("utf-8"))
    else:
      names.append(str(value))
  return names


def _quat_normalize(q: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(q, axis=-1, keepdims=True)
  norm = np.clip(norm, 1.0e-12, None)
  return q / norm


def _root_up_z_from_wxyz(root_quat_wxyz: np.ndarray) -> np.ndarray:
  """Return world z-component of root local +Z axis for quats in wxyz."""
  q = _quat_normalize(root_quat_wxyz)
  # Rotate v=[0,0,1] using quaternion vector formula:
  # v' = v + w*(2*qv×v) + qv×(2*qv×v)
  qv = q[..., 1:]
  v = np.zeros(qv.shape, dtype=q.dtype)
  v[..., 2] = 1.0
  t = 2.0 * np.cross(qv, v)
  v_prime = v + q[..., :1] * t + np.cross(qv, t)
  return v_prime[..., 2]


def _longest_true_run(mask: np.ndarray) -> tuple[int, int | None]:
  if mask.ndim != 1:
    raise ValueError(f"Expected 1D mask, got shape {mask.shape}")
  if not np.any(mask):
    return 0, None
  idx = np.flatnonzero(mask)
  if idx.size == 1:
    return 1, int(idx[0])
  diffs = np.diff(idx)
  # run breaks where diff != 1
  break_points = np.where(diffs != 1)[0]
  starts = np.concatenate(([0], break_points + 1))
  ends = np.concatenate((break_points, [idx.size - 1]))
  lengths = ends - starts + 1
  best_i = int(np.argmax(lengths))
  best_len = int(lengths[best_i])
  first_frame = int(idx[starts[best_i]])
  return best_len, first_frame


def _check_file(path: Path, args: Args) -> FileResult:
  rel = str(path.relative_to(args.npz_root))
  result = FileResult(rel_path=rel, status="ok")

  with np.load(path, allow_pickle=False) as npz:
    required = {"body_quat_w", "body_names"}
    missing = sorted(required.difference(set(npz.files)))
    if missing:
      result.status = "error"
      result.message = f"missing keys: {missing}"
      return result

    body_quat_w = np.asarray(npz["body_quat_w"], dtype=np.float64)
    body_names = _decode_names(np.asarray(npz["body_names"]))

  if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
    result.status = "error"
    result.message = f"invalid body_quat_w shape: {body_quat_w.shape}"
    return result
  result.num_frames = int(body_quat_w.shape[0])
  if result.num_frames == 0:
    result.status = "error"
    result.message = "empty motion (0 frames)"
    return result

  if args.root_body_name not in body_names:
    result.status = "error"
    result.message = f"root body `{args.root_body_name}` not found"
    return result
  root_idx = body_names.index(args.root_body_name)

  up_z = _root_up_z_from_wxyz(body_quat_w[:, root_idx])
  abs_up_z = np.abs(up_z)
  result.min_abs_up_z = float(np.min(abs_up_z))

  horizontal_mask = abs_up_z <= args.horizontal_abs_up_z_threshold
  longest_run, first_frame = _longest_true_run(horizontal_mask)
  result.longest_horizontal_run = longest_run
  result.first_horizontal_frame = first_frame

  if longest_run >= args.min_consecutive_frames:
    result.status = "flagged"
    result.message = (
      f"horizontal run={longest_run} first_frame={first_frame} "
      f"min_abs_up_z={result.min_abs_up_z:.6g}"
    )

  return result


def _apply_action(path: Path, result: FileResult, args: Args) -> None:
  if result.status != "flagged":
    return
  if not args.delete:
    result.action = "dry-run"
    return

  if args.trash_dir is not None:
    rel = path.relative_to(args.npz_root)
    dst = args.trash_dir / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(dst))
    result.action = f"moved:{dst}"
  else:
    path.unlink()
    result.action = "deleted"


def main() -> None:
  args = _parse_args()
  if not args.npz_root.exists():
    raise FileNotFoundError(f"NPZ root does not exist: {args.npz_root}")
  if args.delete and args.trash_dir is not None:
    args.trash_dir.mkdir(parents=True, exist_ok=True)

  npz_files = sorted(args.npz_root.rglob("*.npz"))
  if args.max_files is not None:
    npz_files = npz_files[: max(0, args.max_files)]

  results: list[FileResult] = []
  for path in tqdm(npz_files, desc="Scanning NPZ motions", unit="file"):
    try:
      result = _check_file(path, args)
      _apply_action(path, result, args)
    except Exception as exc:  # noqa: BLE001
      result = FileResult(
        rel_path=str(path.relative_to(args.npz_root)),
        status="error",
        message=f"{type(exc).__name__}: {exc}",
      )
    results.append(result)

  ok_count = sum(r.status == "ok" for r in results)
  flagged_count = sum(r.status == "flagged" for r in results)
  err_count = sum(r.status == "error" for r in results)
  deleted_count = sum(r.action == "deleted" for r in results)
  moved_count = sum(r.action.startswith("moved:") for r in results)
  dry_run_count = sum(r.action == "dry-run" for r in results)

  print(f"Total files scanned: {len(results)}")
  print(f"OK: {ok_count}")
  print(f"Flagged: {flagged_count}")
  print(f"Errors: {err_count}")
  print(
    f"Actions -> deleted: {deleted_count}, moved: {moved_count}, dry-run: {dry_run_count}"
  )

  flagged = [r for r in results if r.status == "flagged"]
  flagged.sort(key=lambda r: (r.min_abs_up_z if r.min_abs_up_z is not None else 1.0))
  if flagged:
    print("\nTop 20 most horizontal flagged motions:")
    for row in flagged[:20]:
      print(
        f"  min_abs_up_z={row.min_abs_up_z:.6g} "
        f"run={row.longest_horizontal_run} "
        f"first={row.first_horizontal_frame} "
        f"path={row.rel_path}"
      )

  errors = [r for r in results if r.status == "error"]
  if errors:
    print("\nFirst 20 errors:")
    for row in errors[:20]:
      print(f"  {row.rel_path}: {row.message}")

  if args.report_json is not None:
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w", encoding="utf-8") as f:
      json.dump(
        {
          "args": {
            "npz_root": str(args.npz_root),
            "root_body_name": args.root_body_name,
            "horizontal_abs_up_z_threshold": args.horizontal_abs_up_z_threshold,
            "min_consecutive_frames": args.min_consecutive_frames,
            "delete": args.delete,
            "trash_dir": str(args.trash_dir) if args.trash_dir is not None else None,
            "max_files": args.max_files,
          },
          "results": [asdict(r) for r in results],
        },
        f,
      )
    print(f"\nWrote JSON report: {args.report_json}")


if __name__ == "__main__":
  main()
