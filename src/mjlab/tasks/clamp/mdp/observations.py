from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs import mdp as env_mdp
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def proprio_history(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Stackable proprioceptive feature vector for temporal encoding.

  The observation manager history buffer is applied on this term via
  ``ObservationTermCfg.history_length``.
  """
  base_lin_vel = env_mdp.builtin_sensor(env, sensor_name="robot/imu_lin_vel")
  base_ang_vel = env_mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel")
  projected_gravity = env_mdp.projected_gravity(env)
  joint_pos = env_mdp.joint_pos_rel(env)
  joint_vel = env_mdp.joint_vel_rel(env)
  actions = env_mdp.last_action(env)
  return torch.cat(
    (
      base_lin_vel,
      base_ang_vel,
      projected_gravity,
      joint_pos,
      joint_vel,
      actions,
    ),
    dim=-1,
  )


def feet_contact_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()
