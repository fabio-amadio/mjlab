"""Launch a Yam RGB showcase run with mined fixed DR settings."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
PolicyName = Literal["naive", "dr"]
ViewerName = Literal["auto", "native", "viser"]


@dataclass(frozen=True)
class ShowcaseSetting:
  cube_half_size: Vec3
  cube_inertia_alpha: float
  cube_rgba: Quat
  cube_pos: Vec3
  cube_yaw: float
  target_pos: Vec3
  camera_pos_delta: Vec3
  camera_rpy_delta: Vec3
  camera_intrinsic_scale: tuple[float, float]
  light_pos_delta: Vec3
  light_dir_delta: Vec3
  gripper_friction: Vec3


SHOWCASE_SETTINGS: Mapping[int, ShowcaseSetting] = {
  # Nominal no-DR baseline. All extra showcase perturbations are fixed to zero
  # or nominal MuJoCo/YAM values.
  0: ShowcaseSetting(
    cube_half_size=(0.02000, 0.02000, 0.02000),
    cube_inertia_alpha=0.0,
    cube_rgba=(0.80000, 0.20000, 0.20000, 1.0),
    cube_pos=(0.30000, 0.00000, 0.03000),
    cube_yaw=0.0,
    target_pos=(0.40000, 0.00000, 0.30000),
    camera_pos_delta=(0.0, 0.0, 0.0),
    camera_rpy_delta=(0.0, 0.0, 0.0),
    camera_intrinsic_scale=(1.0, 1.0),
    light_pos_delta=(0.0, 0.0, 0.0),
    light_dir_delta=(0.0, 0.0, 0.0),
    gripper_friction=(1.0, 0.005, 0.0005),
  ),
  # Naive policy fails with a grasp-like failure; DR policy succeeds.
  487: ShowcaseSetting(
    cube_half_size=(0.02590, 0.01960, 0.02071),
    cube_inertia_alpha=-0.07429,
    cube_rgba=(0.58510, 0.22072, 0.64338, 1.0),
    cube_pos=(0.39329, 0.14511, 0.04317),
    cube_yaw=-2.23572,
    target_pos=(0.35664, 0.19098, 0.38295),
    camera_pos_delta=(0.00147, 0.00184, -0.00273),
    camera_rpy_delta=(-0.01628, 0.00023, 0.02168),
    camera_intrinsic_scale=(0.97568, 1.04608),
    light_pos_delta=(-0.08238, 0.11625, 0.12566),
    light_dir_delta=(0.04244, -0.01276, 0.11406),
    gripper_friction=(0.83834, 0.00275, 0.00055),
  ),
  934: ShowcaseSetting(
    cube_half_size=(0.02540, 0.02378, 0.02474),
    cube_inertia_alpha=0.14070,
    cube_rgba=(0.12841, 0.18529, 0.49226, 1.0),
    cube_pos=(0.27027, -0.15719, 0.03569),
    cube_yaw=2.65117,
    target_pos=(0.39455, 0.13459, 0.37045),
    camera_pos_delta=(-0.00281, 0.00447, 0.00152),
    camera_rpy_delta=(0.00468, 0.01736, 0.02826),
    camera_intrinsic_scale=(1.01662, 1.02800),
    light_pos_delta=(0.01196, 0.07411, -0.11745),
    light_dir_delta=(0.00307, -0.04643, -0.04185),
    gripper_friction=(0.82282, 0.00518, 0.00069),
  ),
  3140: ShowcaseSetting(
    cube_half_size=(0.02470, 0.02090, 0.02370),
    cube_inertia_alpha=0.15168,
    cube_rgba=(0.83972, 0.48962, 0.56613, 1.0),
    cube_pos=(0.25168, 0.10869, 0.04157),
    cube_yaw=1.43135,
    target_pos=(0.47612, -0.14207, 0.33047),
    camera_pos_delta=(-0.00309, -0.00238, -0.00279),
    camera_rpy_delta=(0.00393, -0.01998, 0.01143),
    camera_intrinsic_scale=(1.04505, 0.95143),
    light_pos_delta=(-0.00574, 0.06209, -0.06920),
    light_dir_delta=(0.12367, -0.13936, 0.05322),
    gripper_friction=(0.89119, 0.00218, 0.00038),
  ),
  4362: ShowcaseSetting(
    cube_half_size=(0.01831, 0.02300, 0.02660),
    cube_inertia_alpha=-0.02538,
    cube_rgba=(0.24667, 0.99980, 0.64819, 1.0),
    cube_pos=(0.20371, -0.01817, 0.04322),
    cube_yaw=2.76203,
    target_pos=(0.41871, 0.15522, 0.37816),
    camera_pos_delta=(0.00125, -0.00206, 0.00209),
    camera_rpy_delta=(-0.00385, 0.01075, -0.02167),
    camera_intrinsic_scale=(1.00706, 0.98212),
    light_pos_delta=(0.06443, -0.09533, 0.04572),
    light_dir_delta=(-0.18893, 0.16495, 0.15685),
    gripper_friction=(0.87440, 0.00357, 0.00074),
  ),
  # Both naive and DR policies succeed. Useful as positive/nominal-ish controls.
  967: ShowcaseSetting(
    cube_half_size=(0.01881, 0.01784, 0.02584),
    cube_inertia_alpha=-0.01175,
    cube_rgba=(0.87162, 0.03128, 0.82082, 1.0),
    cube_pos=(0.24719, -0.17096, 0.03023),
    cube_yaw=2.18990,
    target_pos=(0.44463, -0.07096, 0.39515),
    camera_pos_delta=(-0.00260, 0.00104, -0.00056),
    camera_rpy_delta=(0.00515, 0.02938, 0.01651),
    camera_intrinsic_scale=(0.97711, 1.03606),
    light_pos_delta=(0.09695, 0.10831, 0.03112),
    light_dir_delta=(-0.12345, 0.18269, -0.03059),
    gripper_friction=(0.80573, 0.00360, 0.00032),
  ),
  2045: ShowcaseSetting(
    cube_half_size=(0.01941, 0.02091, 0.01966),
    cube_inertia_alpha=-0.09973,
    cube_rgba=(0.83517, 0.34232, 0.18652, 1.0),
    cube_pos=(0.36150, -0.07086, 0.04569),
    cube_yaw=-0.04634,
    target_pos=(0.42140, -0.15939, 0.39449),
    camera_pos_delta=(0.00094, -0.00243, 0.00023),
    camera_rpy_delta=(0.00136, -0.00002, -0.01871),
    camera_intrinsic_scale=(1.02263, 1.04612),
    light_pos_delta=(0.05953, -0.04603, 0.10365),
    light_dir_delta=(0.05700, -0.06260, 0.01781),
    gripper_friction=(0.76306, 0.00397, 0.00150),
  ),
  1833: ShowcaseSetting(
    cube_half_size=(0.01900, 0.02043, 0.01940),
    cube_inertia_alpha=0.05342,
    cube_rgba=(0.29836, 0.20089, 0.07868, 1.0),
    cube_pos=(0.29374, 0.19409, 0.04789),
    cube_yaw=-2.99928,
    target_pos=(0.44360, -0.14427, 0.34979),
    camera_pos_delta=(0.00483, -0.00597, 0.00139),
    camera_rpy_delta=(-0.00226, -0.02906, -0.00979),
    camera_intrinsic_scale=(1.00016, 1.04092),
    light_pos_delta=(0.04010, 0.10346, -0.11232),
    light_dir_delta=(-0.03358, 0.04454, -0.18803),
    gripper_friction=(0.99222, 0.00372, 0.00024),
  ),
  4801: ShowcaseSetting(
    cube_half_size=(0.02659, 0.01779, 0.01642),
    cube_inertia_alpha=-0.18505,
    cube_rgba=(0.58831, 0.18740, 0.33655, 1.0),
    cube_pos=(0.29480, -0.00251, 0.04424),
    cube_yaw=1.17762,
    target_pos=(0.39563, -0.10321, 0.37688),
    camera_pos_delta=(0.00573, 0.00060, -0.00194),
    camera_rpy_delta=(-0.00730, 0.01578, 0.00660),
    camera_intrinsic_scale=(0.96181, 1.02960),
    light_pos_delta=(-0.00324, -0.05956, -0.02941),
    light_dir_delta=(-0.15120, 0.06669, -0.11065),
    gripper_friction=(0.75271, 0.00106, 0.00040),
  ),
}


@dataclass(frozen=True)
class ShowcasePlayConfig:
  policy: PolicyName = "naive"
  """Which policy checkpoint to launch."""

  candidate: int = 934
  """Mined candidate condition id."""

  naive_checkpoint_file: str = (
    "logs/rsl_rl/yam_lift_cube_rgb_dr/wandb_checkpoints/trixiwny/model_2999.pt"
  )
  """Checkpoint for the naive policy."""

  dr_checkpoint_file: str = (
    "logs/rsl_rl/yam_lift_cube_rgb_dr/wandb_checkpoints/ci0e7hls/model_3999.pt"
  )
  """Checkpoint for the DR policy."""

  env_task: str = "Mjlab-Lift-Cube-Yam-Rgb-Dr"
  """Environment task used for visualization."""

  naive_policy_task: str = "Mjlab-Lift-Cube-Yam-Rgb-Naive"
  """Task whose runner config matches the naive checkpoint."""

  dr_policy_task: str = "Mjlab-Lift-Cube-Yam-Rgb-Dr"
  """Task whose runner config matches the DR checkpoint."""

  num_envs: int = 1
  """Number of parallel envs to launch. One is recommended for the showcase."""

  seed: int = 1
  """Seed for deterministic reset and viewer behavior."""

  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""

  viewer: ViewerName = "auto"
  """Viewer backend."""

  no_terminations: bool = False
  """Disable terminations if you want to keep watching after failures."""


def _fixed_vec3(value: Vec3) -> dict[int, tuple[float, float]]:
  return {axis: (v, v) for axis, v in enumerate(value)}


def _fixed_rgba(value: Quat) -> dict[int, tuple[float, float]]:
  return {axis: (v, v) for axis, v in enumerate(value)}


def _apply_fixed_setting(env_cfg, setting: ShowcaseSetting) -> None:
  env_cfg.events["cube_size"].params["ranges"] = _fixed_vec3(setting.cube_half_size)
  env_cfg.events["cube_inertia"].params["alpha_range"] = (
    setting.cube_inertia_alpha,
    setting.cube_inertia_alpha,
  )
  env_cfg.events["cube_color"].params["ranges"] = _fixed_rgba(setting.cube_rgba)
  env_cfg.events["camera_pos"].params["ranges"] = _fixed_vec3(setting.camera_pos_delta)
  env_cfg.events["camera_quat"].params["roll_range"] = (
    setting.camera_rpy_delta[0],
    setting.camera_rpy_delta[0],
  )
  env_cfg.events["camera_quat"].params["pitch_range"] = (
    setting.camera_rpy_delta[1],
    setting.camera_rpy_delta[1],
  )
  env_cfg.events["camera_quat"].params["yaw_range"] = (
    setting.camera_rpy_delta[2],
    setting.camera_rpy_delta[2],
  )
  env_cfg.events["camera_intrinsic"].params["ranges"] = {
    0: (setting.camera_intrinsic_scale[0], setting.camera_intrinsic_scale[0]),
    1: (setting.camera_intrinsic_scale[1], setting.camera_intrinsic_scale[1]),
  }
  env_cfg.events["light_pos"].params["ranges"] = _fixed_vec3(setting.light_pos_delta)
  env_cfg.events["light_dir"].params["ranges"] = _fixed_vec3(setting.light_dir_delta)

  friction_events = (
    ("fingertip_friction_slide", setting.gripper_friction[0]),
    ("fingertip_friction_spin", setting.gripper_friction[1]),
    ("fingertip_friction_roll", setting.gripper_friction[2]),
  )
  for event_name, value in friction_events:
    if event_name in env_cfg.events:
      env_cfg.events[event_name].params["ranges"] = (value, value)

  command = env_cfg.commands["lift_height"]
  command.resampling_time_range = (1.0e9, 1.0e9)
  command.difficulty = "dynamic"
  command.target_position_range.x = (setting.target_pos[0], setting.target_pos[0])
  command.target_position_range.y = (setting.target_pos[1], setting.target_pos[1])
  command.target_position_range.z = (setting.target_pos[2], setting.target_pos[2])
  command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(setting.cube_pos[0], setting.cube_pos[0]),
    y=(setting.cube_pos[1], setting.cube_pos[1]),
    z=(setting.cube_pos[2], setting.cube_pos[2]),
    yaw=(setting.cube_yaw, setting.cube_yaw),
  )


def _resolve_policy_cfg(cfg: ShowcasePlayConfig) -> tuple[str, str]:
  if cfg.policy == "naive":
    return cfg.naive_policy_task, cfg.naive_checkpoint_file
  return cfg.dr_policy_task, cfg.dr_checkpoint_file


def run_showcase_play(cfg: ShowcasePlayConfig) -> None:
  configure_torch_backends()
  if cfg.candidate not in SHOWCASE_SETTINGS:
    raise ValueError(
      f"Unknown candidate {cfg.candidate}. Supported: {sorted(SHOWCASE_SETTINGS)}"
    )

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  policy_task, checkpoint_file = _resolve_policy_cfg(cfg)
  checkpoint_path = Path(checkpoint_file)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

  env_cfg = load_env_cfg(cfg.env_task, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  if cfg.no_terminations:
    env_cfg.terminations = {}
  _apply_fixed_setting(env_cfg, SHOWCASE_SETTINGS[cfg.candidate])

  agent_cfg = load_rl_cfg(policy_task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(policy_task) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = cfg.viewer

  print(
    f"[INFO] Launching candidate={cfg.candidate} policy={cfg.policy} "
    f"viewer={resolved_viewer}"
  )
  if resolved_viewer == "native":
    NativeMujocoViewer(wrapped_env, policy).run()
  else:
    ViserPlayViewer(wrapped_env, policy).run()

  wrapped_env.close()


def main() -> None:
  import mjlab.tasks  # noqa: F401

  cfg = tyro.cli(
    ShowcasePlayConfig,
    args=sys.argv[1:],
    prog=sys.argv[0],
    config=mjlab.TYRO_FLAGS,
  )
  run_showcase_play(cfg)


if __name__ == "__main__":
  main()
