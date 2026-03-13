# CLAMP Motion

Internal implementation of CLAMP motion commands.

External code should usually import command types from:

- `mjlab.tasks.clamp.mdp`
- `mjlab.tasks.clamp.mdp.commands`

This folder holds the implementation behind those exports.

## Files

- `base.py`
  Shared motion state, frame queries, metrics, and debug hooks.

- `joint_ref.py`
  Single-step student command: current `joint_pos` and `joint_vel`.

- `future_joint_ref.py`
  Future-stacked command variants used by the teacher.

- `teacher_student.py`
  Distillation command. Student and teacher representations come from the same motion state.

- `representations.py`
  Tensor serialization helpers for the concrete command layouts.

- `sampling.py`
  Resampling and per-step update logic.

- `library.py`
  Motion loading and multi-clip library queries.

- `indexing.py`
  Body-name resolution and mapping checks.

- `debug_visualizer.py`
  Ghost visualization of the current reference.

## Runtime Shape

The split is:

- `base.py`: shared motion state
- `representations.py`: what goes into each command tensor
- `sampling.py`: how motion state is sampled and advanced

Concrete command classes mostly choose a representation on top of the shared state.

## Distillation

`TeacherStudentMotionCommand` keeps teacher and student synchronized by generating both representations from one motion state:

- `"default"`: student view
- `"teacher"`: future teacher view

This is the reason teacher and student should not be separate command terms.

## Useful Entry Points

If you are reading the code for the first time:

1. `base.py`
2. `joint_ref.py`
3. `future_joint_ref.py`
4. `teacher_student.py`
5. `sampling.py`

If you add a new command variant, the usual work is:

1. add a representation helper in `representations.py` if needed
2. add a new `MotionCommand` subclass
3. override `future_sampling_step_offsets` if the command needs future frames
4. add a cfg class and re-export it from `mdp/commands.py` if it should be public
