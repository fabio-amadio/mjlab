from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.tasks.clamp.mdp.student_commands import TeacherStudentMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def feet_contact_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def motion_student_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if isinstance(command, TeacherStudentMotionCommand):
    return command.student_command
  fallback = env.command_manager.get_command(command_name)
  assert fallback is not None
  return fallback


def motion_teacher_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if isinstance(command, TeacherStudentMotionCommand):
    return command.teacher_command
  fallback = env.command_manager.get_command(command_name)
  assert fallback is not None
  return fallback
