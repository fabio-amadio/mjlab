"""Headless Yam RGB DR sweep for finding showcase failure settings.

The script samples randomized evaluation conditions from the DR environment and
evaluates both the naive and DR policies on the same conditions. Each CSV row is
one condition:

  DR settings | naive policy metrics | DR policy metrics

Failure labels are behavioral heuristics:

* ``grasp_failure`` means the end-effector approached the cube or lifted it a
  little, but the episode never reached the goal.
* ``vision_failure`` means the end-effector never approached the cube and the
  cube was not lifted, which is a proxy for "the policy did not locate the box".
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

PolicyFn = Callable[[Any], torch.Tensor]


@dataclass(frozen=True)
class ShowcaseEvalConfig:
  naive_checkpoint_file: str
  """Checkpoint for the naive policy."""

  dr_checkpoint_file: str
  """Checkpoint for the DR policy."""

  env_task: str = "Mjlab-Lift-Cube-Yam-Rgb-Dr"
  """DR environment used to sample test conditions."""

  naive_policy_task: str = "Mjlab-Lift-Cube-Yam-Rgb-Naive"
  """Task whose runner config matches the naive checkpoint."""

  dr_policy_task: str = "Mjlab-Lift-Cube-Yam-Rgb-Dr"
  """Task whose runner config matches the DR checkpoint."""

  num_conditions: int = 4096
  """Total number of randomized DR conditions to evaluate."""

  num_envs: int = 256
  """Number of parallel conditions per batch."""

  episode_length_s: float | None = None
  """Optional episode length override. Leave unset to use the task default."""

  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""

  seed: int = 1
  """Base random seed. Each batch adds its index to this seed."""

  disable_cube_color_dr: bool = False
  """Disable cube color DR if you want to isolate shape/inertia/camera/light."""

  output_file: str = "showcase_dr_sweep.csv"
  """Per-condition CSV dataset output path."""

  summary_file: str = "showcase_dr_sweep_summary.json"
  """Aggregate JSON summary output path."""

  grasp_distance_threshold: float = 0.065
  """EE/cube distance below which a failed episode is counted as a grasp attempt."""

  approach_distance_threshold: float = 0.10
  """If the EE never gets this close, a failure is counted as vision/localization."""

  lift_attempt_threshold: float = 0.008
  """Cube lift above reset height that counts as an attempted grasp."""


@dataclass(frozen=True)
class PolicyEvalSpec:
  name: str
  task: str
  checkpoint_file: str


@dataclass
class EpisodeTrackers:
  initialized: torch.Tensor
  initial_cube_z: torch.Tensor
  min_ee_cube_dist: torch.Tensor
  min_goal_dist: torch.Tensor
  max_cube_lift: torch.Tensor
  max_cube_height: torch.Tensor
  steps: torch.Tensor

  @classmethod
  def create(cls, num_envs: int, device: str | torch.device) -> EpisodeTrackers:
    return cls(
      initialized=torch.zeros(num_envs, dtype=torch.bool, device=device),
      initial_cube_z=torch.zeros(num_envs, device=device),
      min_ee_cube_dist=torch.full((num_envs,), torch.inf, device=device),
      min_goal_dist=torch.full((num_envs,), torch.inf, device=device),
      max_cube_lift=torch.zeros(num_envs, device=device),
      max_cube_height=torch.zeros(num_envs, device=device),
      steps=torch.zeros(num_envs, dtype=torch.long, device=device),
    )


def _to_list(tensor: torch.Tensor) -> list[Any]:
  return tensor.detach().cpu().tolist()


def _add_vector(
  row: dict[str, Any],
  prefix: str,
  values: list[float],
  names: tuple[str, ...],
) -> None:
  for name, value in zip(names, values, strict=True):
    row[f"{prefix}_{name}"] = value


def _add_optional_vector(
  row: dict[str, Any],
  prefix: str,
  values: list[float] | None,
  names: tuple[str, ...],
) -> None:
  if values is None:
    for name in names:
      row[f"{prefix}_{name}"] = None
    return
  _add_vector(row, prefix, values, names)


def _named_id(names: tuple[str, ...], ids: torch.Tensor, name: str) -> int | None:
  if name not in names:
    return None
  return int(ids[names.index(name)].item())


def _batched_field(
  field: torch.Tensor,
  entity_id: int | None,
) -> list[list[float]] | None:
  if entity_id is None:
    return None
  return _to_list(field[:, entity_id])


def _row_values(values: list[list[float]] | None, env_id: int) -> list[float] | None:
  if values is None:
    return None
  return values[env_id]


def _fingertip_geom_ids(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  pattern = re.compile(r"[lr]f_down(6|7|8|9|10|11)_collision")
  local_ids = [
    idx for idx, name in enumerate(robot.geom_names) if pattern.fullmatch(name)
  ]
  return robot.indexing.geom_ids[local_ids]


def _collect_condition_rows(
  env: ManagerBasedRlEnv,
  seed: int,
  condition_offset: int,
) -> list[dict[str, Any]]:
  cube = env.scene["cube"]
  robot = env.scene["robot"]
  model = env.sim.model
  command = cast(Any, env.command_manager.get_term("lift_height"))

  cube_geom_id = _named_id(cube.geom_names, cube.indexing.geom_ids, "cube_geom")
  cube_body_id = _named_id(cube.body_names, cube.indexing.body_ids, "cube")
  camera_id = _named_id(robot.camera_names, robot.indexing.cam_ids, "camera_d405")
  light_id = _named_id(robot.light_names, robot.indexing.light_ids, "spotlight")

  cube_sizes = _batched_field(model.geom_size, cube_geom_id)
  cube_colors = _batched_field(model.geom_rgba, cube_geom_id)
  cube_masses = _batched_field(model.body_mass.unsqueeze(-1), cube_body_id)
  cube_inertias = _batched_field(model.body_inertia, cube_body_id)
  cube_ipos = _batched_field(model.body_ipos, cube_body_id)
  cube_iquats = _batched_field(model.body_iquat, cube_body_id)
  camera_pos = _batched_field(model.cam_pos, camera_id)
  camera_quat = _batched_field(model.cam_quat, camera_id)
  camera_intrinsic = _batched_field(model.cam_intrinsic, camera_id)
  light_pos = _batched_field(model.light_pos, light_id)
  light_dir = _batched_field(model.light_dir, light_id)
  initial_cube_pos = _to_list(cube.data.root_link_pos_w)
  target_pos = _to_list(command.target_pos)
  fingertip_friction = model.geom_friction[:, _fingertip_geom_ids(env)]
  friction_mean = _to_list(fingertip_friction.mean(dim=1))
  friction_min = _to_list(fingertip_friction.min(dim=1).values)
  friction_max = _to_list(fingertip_friction.max(dim=1).values)

  rows = []
  for env_id in range(env.num_envs):
    row: dict[str, Any] = {
      "condition_id": condition_offset + env_id,
      "batch_seed": seed,
      "env_id": env_id,
    }
    _add_optional_vector(
      row, "cube_half_size", _row_values(cube_sizes, env_id), ("x", "y", "z")
    )
    cube_mass = _row_values(cube_masses, env_id)
    row["cube_mass"] = None if cube_mass is None else cube_mass[0]
    _add_optional_vector(
      row, "cube_inertia", _row_values(cube_inertias, env_id), ("x", "y", "z")
    )
    _add_optional_vector(
      row, "cube_ipos", _row_values(cube_ipos, env_id), ("x", "y", "z")
    )
    _add_optional_vector(
      row, "cube_iquat", _row_values(cube_iquats, env_id), ("w", "x", "y", "z")
    )
    _add_optional_vector(
      row, "cube_rgba", _row_values(cube_colors, env_id), ("r", "g", "b", "a")
    )
    _add_optional_vector(
      row, "camera_pos", _row_values(camera_pos, env_id), ("x", "y", "z")
    )
    _add_optional_vector(
      row, "camera_quat", _row_values(camera_quat, env_id), ("w", "x", "y", "z")
    )
    _add_optional_vector(
      row,
      "camera_intrinsic",
      _row_values(camera_intrinsic, env_id),
      ("fx", "fy", "cx", "cy"),
    )
    _add_optional_vector(
      row, "light_pos", _row_values(light_pos, env_id), ("x", "y", "z")
    )
    _add_optional_vector(
      row, "light_dir", _row_values(light_dir, env_id), ("x", "y", "z")
    )
    _add_vector(
      row,
      "gripper_friction_mean",
      friction_mean[env_id],
      ("slide", "spin", "roll"),
    )
    _add_vector(
      row,
      "gripper_friction_min",
      friction_min[env_id],
      ("slide", "spin", "roll"),
    )
    _add_vector(
      row,
      "gripper_friction_max",
      friction_max[env_id],
      ("slide", "spin", "roll"),
    )
    _add_vector(row, "initial_cube_pos", initial_cube_pos[env_id], ("x", "y", "z"))
    _add_vector(row, "target_pos", target_pos[env_id], ("x", "y", "z"))
    rows.append(row)
  return rows


def _prepare_env_cfg(
  cfg: ShowcaseEvalConfig,
  num_envs: int,
  seed: int,
) -> Any:
  env_cfg = load_env_cfg(cfg.env_task, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  if cfg.episode_length_s is not None:
    env_cfg.episode_length_s = cfg.episode_length_s
  if cfg.disable_cube_color_dr:
    env_cfg.events.pop("cube_color", None)

  command = env_cfg.commands.get("lift_height")
  if command is None:
    raise ValueError(f"Task '{cfg.env_task}' has no lift_height command.")
  command.resampling_time_range = (1.0e9, 1.0e9)
  return env_cfg


def _load_policy(
  env: RslRlVecEnvWrapper,
  task: str,
  checkpoint_file: str,
  device: str,
) -> PolicyFn:
  agent_cfg = load_rl_cfg(task)
  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    checkpoint_file,
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device)


def _update_trackers(
  env: ManagerBasedRlEnv,
  trackers: EpisodeTrackers,
  active: torch.Tensor,
) -> None:
  env_ids = active.nonzero(as_tuple=False).squeeze(-1)
  if len(env_ids) == 0:
    return

  robot = env.scene["robot"]
  cube = env.scene["cube"]
  command = cast(Any, env.command_manager.get_term("lift_height"))

  site_idx = robot.site_names.index("grasp_site")
  ee_pos = robot.data.site_pos_w[:, site_idx]
  cube_pos = cube.data.root_link_pos_w
  target_pos = command.target_pos

  cube_z = cube_pos[:, 2]
  not_initialized = active & ~trackers.initialized
  if not_initialized.any():
    trackers.initial_cube_z[not_initialized] = cube_z[not_initialized]
    trackers.initialized[not_initialized] = True

  ee_cube_dist = torch.norm(cube_pos - ee_pos, dim=-1)
  goal_dist = torch.norm(target_pos - cube_pos, dim=-1)
  cube_lift = cube_z - trackers.initial_cube_z

  trackers.min_ee_cube_dist[env_ids] = torch.minimum(
    trackers.min_ee_cube_dist[env_ids], ee_cube_dist[env_ids]
  )
  trackers.min_goal_dist[env_ids] = torch.minimum(
    trackers.min_goal_dist[env_ids], goal_dist[env_ids]
  )
  trackers.max_cube_lift[env_ids] = torch.maximum(
    trackers.max_cube_lift[env_ids], cube_lift[env_ids]
  )
  trackers.max_cube_height[env_ids] = torch.maximum(
    trackers.max_cube_height[env_ids], cube_z[env_ids]
  )
  trackers.steps[env_ids] += 1


def _metric_rows(
  cfg: ShowcaseEvalConfig,
  env_ids: torch.Tensor,
  trackers: EpisodeTrackers,
  terminated: torch.Tensor,
  timed_out: torch.Tensor,
  success_threshold: float,
) -> list[dict[str, Any]]:
  rows = []
  for env_id in env_ids.tolist():
    min_dist = trackers.min_ee_cube_dist[env_id].item()
    min_goal = trackers.min_goal_dist[env_id].item()
    max_lift = trackers.max_cube_lift[env_id].item()
    success = min_goal <= success_threshold
    grasp_attempt = (
      min_dist <= cfg.grasp_distance_threshold or max_lift >= cfg.lift_attempt_threshold
    )
    vision_failure = (
      min_dist > cfg.approach_distance_threshold
      and max_lift < cfg.lift_attempt_threshold
    )
    if success:
      failure_mode = "success"
    elif grasp_attempt:
      failure_mode = "grasp_failure"
    elif vision_failure:
      failure_mode = "vision_failure"
    elif terminated[env_id].item():
      failure_mode = "termination_failure"
    else:
      failure_mode = "other_failure"

    rows.append(
      {
        "success": bool(success),
        "failure_mode": failure_mode,
        "terminated": bool(terminated[env_id].item()),
        "timed_out": bool(timed_out[env_id].item()),
        "steps": int(trackers.steps[env_id].item()),
        "min_ee_cube_dist": min_dist,
        "min_goal_dist": min_goal,
        "max_cube_lift": max_lift,
        "max_cube_height": trackers.max_cube_height[env_id].item(),
      }
    )
  return rows


def _prefix_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
  return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _run_policy_batch(
  cfg: ShowcaseEvalConfig,
  policy_spec: PolicyEvalSpec,
  policy: PolicyFn | None,
  seed: int,
  num_envs: int,
  condition_offset: int,
) -> tuple[PolicyFn, list[dict[str, Any]], list[dict[str, Any]]]:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = _prepare_env_cfg(cfg, num_envs, seed)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped_env = RslRlVecEnvWrapper(
    env,
    clip_actions=load_rl_cfg(policy_spec.task).clip_actions,
  )
  if policy is None:
    policy = _load_policy(
      wrapped_env, policy_spec.task, policy_spec.checkpoint_file, device
    )

  condition_rows = _collect_condition_rows(
    wrapped_env.unwrapped,
    seed=seed,
    condition_offset=condition_offset,
  )
  command = cast(Any, wrapped_env.unwrapped.command_manager.get_term("lift_height"))
  success_threshold = float(command.cfg.success_threshold)
  trackers = EpisodeTrackers.create(num_envs, wrapped_env.device)
  active = torch.ones(num_envs, dtype=torch.bool, device=wrapped_env.device)
  metrics_by_env: list[dict[str, Any] | None] = [None] * num_envs

  obs = wrapped_env.get_observations()
  max_steps = wrapped_env.max_episode_length + 1
  for _ in range(max_steps):
    if not active.any():
      break
    _update_trackers(wrapped_env.unwrapped, trackers, active)
    with torch.no_grad():
      actions = policy(obs)
    actions = actions.clone()
    actions[~active] = 0.0
    obs, _, dones, _ = wrapped_env.step(actions)

    done_ids = (dones.bool() & active).nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) == 0:
      continue

    terminated = wrapped_env.unwrapped.reset_terminated.clone()
    timed_out = wrapped_env.unwrapped.reset_time_outs.clone()
    metric_rows = _metric_rows(
      cfg,
      done_ids,
      trackers,
      terminated,
      timed_out,
      success_threshold,
    )
    for env_id, metric_row in zip(done_ids.tolist(), metric_rows, strict=True):
      metrics_by_env[env_id] = metric_row
    active[done_ids] = False

  remaining_ids = active.nonzero(as_tuple=False).squeeze(-1)
  if len(remaining_ids) > 0:
    false_flags = torch.zeros(num_envs, dtype=torch.bool, device=wrapped_env.device)
    metric_rows = _metric_rows(
      cfg,
      remaining_ids,
      trackers,
      false_flags,
      false_flags,
      success_threshold,
    )
    for env_id, metric_row in zip(remaining_ids.tolist(), metric_rows, strict=True):
      metric_row["failure_mode"] = "max_step_failure"
      metrics_by_env[env_id] = metric_row

  wrapped_env.close()
  return policy, condition_rows, [cast(dict[str, Any], row) for row in metrics_by_env]


def _rate(rows: list[dict[str, Any]], key: str, value: Any) -> float:
  if not rows:
    return 0.0
  return sum(1 for row in rows if row[key] == value) / len(rows)


def _policy_summary(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
  metrics = [
    {key.removeprefix(f"{prefix}_"): value for key, value in row.items()}
    for row in rows
    if f"{prefix}_success" in row
  ]
  success_rate = _rate(metrics, "success", True)
  return {
    "success_rate": success_rate,
    "grasp_failure_rate": _rate(metrics, "failure_mode", "grasp_failure"),
    "vision_failure_rate": _rate(metrics, "failure_mode", "vision_failure"),
    "other_failure_rate": 1.0
    - success_rate
    - _rate(metrics, "failure_mode", "grasp_failure")
    - _rate(metrics, "failure_mode", "vision_failure"),
  }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  naive = _policy_summary(rows, "naive")
  robust = _policy_summary(rows, "dr")
  naive_fail_dr_success = [
    row for row in rows if (not row["naive_success"]) and row["dr_success"]
  ]
  naive_grasp_fail_dr_success = [
    row
    for row in rows
    if row["naive_failure_mode"] == "grasp_failure" and row["dr_success"]
  ]
  naive_vision_fail_dr_success = [
    row
    for row in rows
    if row["naive_failure_mode"] == "vision_failure" and row["dr_success"]
  ]
  return {
    "conditions": len(rows),
    "naive": naive,
    "dr": robust,
    "dr_minus_naive_success_rate": robust["success_rate"] - naive["success_rate"],
    "naive_fail_dr_success_count": len(naive_fail_dr_success),
    "naive_grasp_fail_dr_success_count": len(naive_grasp_fail_dr_success),
    "naive_vision_fail_dr_success_count": len(naive_vision_fail_dr_success),
  }


def _print_summary(summary: dict[str, Any]) -> None:
  print("\nShowcase DR Sweep Summary")
  print("=" * 80)
  print(f"conditions: {summary['conditions']}")
  for name in ("naive", "dr"):
    item = summary[name]
    print(
      f"{name:<6} success={item['success_rate']:.3f} "
      f"grasp_fail={item['grasp_failure_rate']:.3f} "
      f"vision_fail={item['vision_failure_rate']:.3f}"
    )
  print(f"dr_minus_naive_success={summary['dr_minus_naive_success_rate']:.3f}")
  print(
    "naive_fail_dr_success="
    f"{summary['naive_fail_dr_success_count']} "
    "naive_grasp_fail_dr_success="
    f"{summary['naive_grasp_fail_dr_success_count']} "
    "naive_vision_fail_dr_success="
    f"{summary['naive_vision_fail_dr_success_count']}"
  )
  print("=" * 80)


def run_showcase_eval(cfg: ShowcaseEvalConfig) -> dict[str, Any]:
  configure_torch_backends()
  for path in (cfg.naive_checkpoint_file, cfg.dr_checkpoint_file):
    if not Path(path).exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")

  naive_spec = PolicyEvalSpec("naive", cfg.naive_policy_task, cfg.naive_checkpoint_file)
  dr_spec = PolicyEvalSpec("dr", cfg.dr_policy_task, cfg.dr_checkpoint_file)
  naive_policy: PolicyFn | None = None
  dr_policy: PolicyFn | None = None
  rows: list[dict[str, Any]] = []

  for condition_offset in range(0, cfg.num_conditions, cfg.num_envs):
    batch_idx = condition_offset // cfg.num_envs
    batch_size = min(cfg.num_envs, cfg.num_conditions - condition_offset)
    batch_seed = cfg.seed + batch_idx
    print(
      f"\n[INFO] Batch {batch_idx}: seed={batch_seed} "
      f"conditions={condition_offset}-{condition_offset + batch_size - 1}"
    )

    naive_policy, condition_rows, naive_metrics = _run_policy_batch(
      cfg, naive_spec, naive_policy, batch_seed, batch_size, condition_offset
    )
    dr_policy, _, dr_metrics = _run_policy_batch(
      cfg, dr_spec, dr_policy, batch_seed, batch_size, condition_offset
    )

    for condition_row, naive_row, dr_row in zip(
      condition_rows, naive_metrics, dr_metrics, strict=True
    ):
      rows.append(
        {
          **condition_row,
          **_prefix_metrics("naive", naive_row),
          **_prefix_metrics("dr", dr_row),
        }
      )

  summary = _summarize(rows)
  result = {"config": asdict(cfg), "summary": summary}
  _print_summary(summary)

  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\n[INFO] Wrote paired DR dataset to {output_path}")

  summary_path = Path(cfg.summary_file)
  summary_path.parent.mkdir(parents=True, exist_ok=True)
  with summary_path.open("w") as f:
    json.dump(result, f, indent=2)
  print(f"[INFO] Wrote summary to {summary_path}")
  return result


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(
    ShowcaseEvalConfig,
    args=sys.argv[1:],
    prog=sys.argv[0],
    config=mjlab.TYRO_FLAGS,
  )
  run_showcase_eval(cfg)


if __name__ == "__main__":
  main()
