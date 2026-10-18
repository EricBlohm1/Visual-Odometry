# Visual Odometry Learning Scaffold

This repository is an intentionally incomplete workspace for learning visual
odometry from scratch with KITTI. It contains guidance, not a working loader or
VO pipeline. Write the data handling, mathematics, algorithms, and plotting
yourself.

Place the KITTI dataset under `data/kitti/`. KITTI odometry and KITTI raw use
different directory layouts; odometry ground truth may also be distributed
separately.

## Modules

- `src/main.py` — ordered roadmap and practical learning tips.
- `src/dataset.py` — KITTI files, frames, timestamps, calibration, and poses.
- `src/camera.py` — intrinsics, coordinate frames, and transformations.
- `src/features.py` — feature detection, descriptors, and correspondences.
- `src/motion.py` — two-view geometry and relative camera motion.
- `src/odometry.py` — pose conventions and trajectory accumulation.
- `src/visualize.py` — trajectory plotting, comparison, and evaluation.

## Suggested order

Follow the numbered roadmap in `src/main.py`: begin with dataset inspection
and image loading, then calibration, features, relative motion, trajectory
accumulation, and finally visualization and evaluation. Validate each stage
independently before moving on.
