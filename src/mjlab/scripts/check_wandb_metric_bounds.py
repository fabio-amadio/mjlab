"""Inspect a W&B metric history for non-finite values and outliers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tyro
import wandb


@dataclass(frozen=True)
class CheckWandbMetricConfig:
  """Configuration for checking metric bounds in a W&B run."""

  wandb_run_path: str
  """W&B run path, e.g. 'entity/project/run_id'."""

  metric: str = "mean_reward"
  """Metric key to inspect."""

  top_k: int = 10
  """How many largest-absolute finite samples to print."""

  max_rows: int | None = None
  """Optional cap on scanned rows (None = scan all)."""

  abs_outlier_threshold: float | None = None
  """Optional absolute threshold to flag outliers, e.g. 1e4."""

  output_json: str | None = None
  """Optional path to save summary JSON."""


def _to_float(value: Any) -> float | None:
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def run_check(cfg: CheckWandbMetricConfig) -> dict[str, Any]:
  api = wandb.Api()
  run = api.run(cfg.wandb_run_path)

  total_rows = 0
  missing_rows = 0
  non_numeric_rows = 0
  non_finite_rows = 0
  finite_points: list[tuple[int | None, float]] = []
  non_finite_examples: list[tuple[int | None, Any]] = []

  for row in run.scan_history(keys=["_step", cfg.metric]):
    total_rows += 1
    if cfg.max_rows is not None and total_rows > cfg.max_rows:
      break

    raw = row.get(cfg.metric)
    step = row.get("_step")
    step_int = int(step) if isinstance(step, (int, np.integer)) else None
    if raw is None:
      missing_rows += 1
      continue

    value = _to_float(raw)
    if value is None:
      non_numeric_rows += 1
      continue

    if not math.isfinite(value):
      non_finite_rows += 1
      if len(non_finite_examples) < 10:
        non_finite_examples.append((step_int, raw))
      continue

    finite_points.append((step_int, value))

  summary: dict[str, Any] = {
    "wandb_run_path": cfg.wandb_run_path,
    "metric": cfg.metric,
    "total_rows_scanned": total_rows,
    "missing_metric_rows": missing_rows,
    "non_numeric_rows": non_numeric_rows,
    "non_finite_rows": non_finite_rows,
    "finite_rows": len(finite_points),
    "non_finite_examples": non_finite_examples,
  }

  print(f"[INFO] Run: {cfg.wandb_run_path}")
  print(f"[INFO] Metric: {cfg.metric}")
  print(
    "[INFO] Rows scanned: "
    f"{total_rows} (finite={len(finite_points)}, missing={missing_rows}, "
    f"non-numeric={non_numeric_rows}, non-finite={non_finite_rows})"
  )

  if len(finite_points) == 0:
    print("[WARN] No finite samples found for this metric.")
  else:
    values = np.asarray([value for _, value in finite_points], dtype=np.float64)
    min_idx = int(np.argmin(values))
    max_idx = int(np.argmax(values))
    min_step, min_val = finite_points[min_idx]
    max_step, max_val = finite_points[max_idx]

    percentiles = {
      "p01": float(np.percentile(values, 1)),
      "p05": float(np.percentile(values, 5)),
      "p50": float(np.percentile(values, 50)),
      "p95": float(np.percentile(values, 95)),
      "p99": float(np.percentile(values, 99)),
    }

    print(f"[INFO] Min: {min_val:.8g} at step={min_step}")
    print(f"[INFO] Max: {max_val:.8g} at step={max_step}")
    print(
      "[INFO] Percentiles: "
      f"p01={percentiles['p01']:.8g}, p05={percentiles['p05']:.8g}, "
      f"p50={percentiles['p50']:.8g}, p95={percentiles['p95']:.8g}, "
      f"p99={percentiles['p99']:.8g}"
    )

    top_k = max(int(cfg.top_k), 0)
    ranked = sorted(finite_points, key=lambda item: abs(item[1]), reverse=True)
    top_abs = ranked[:top_k]
    if top_abs:
      print(f"[INFO] Top {len(top_abs)} |value| samples:")
      for step, value in top_abs:
        print(f"  step={step}, value={value:.8g}, abs={abs(value):.8g}")

    outlier_count = None
    if cfg.abs_outlier_threshold is not None:
      threshold = abs(float(cfg.abs_outlier_threshold))
      outlier_mask = np.abs(values) > threshold
      outlier_count = int(outlier_mask.sum())
      print(
        f"[INFO] |value| > {threshold:.8g}: {outlier_count}/{len(values)} finite samples"
      )

    summary.update(
      {
        "min_value": float(min_val),
        "min_step": min_step,
        "max_value": float(max_val),
        "max_step": max_step,
        "percentiles": percentiles,
        "top_abs_samples": [
          {"step": step, "value": float(value), "abs_value": abs(float(value))}
          for step, value in top_abs
        ],
        "abs_outlier_threshold": cfg.abs_outlier_threshold,
        "abs_outlier_count": outlier_count,
      }
    )

  if cfg.output_json is not None:
    out_path = Path(cfg.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
    print(f"[INFO] Wrote summary JSON to {out_path}")

  return summary


def main() -> None:
  cfg = tyro.cli(
    CheckWandbMetricConfig,
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_check(cfg)


if __name__ == "__main__":
  main()
