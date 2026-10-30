# Monocular Visual Odometry

This repository implements a basic monocular visual-odometry pipeline for the
KITTI odometry dataset. It detects and matches image features, estimates
relative motion from consecutive left-camera frames with an essential matrix,
and accumulates the estimated poses into a trajectory.

The current motion estimate is monocular: translation has arbitrary scale, so
the plotted trajectory is for visual comparison and is not a metric estimate.
The implementation does not include loop closure or drift correction. Failed
motion estimates are represented by `NaN` poses in the trajectory; the
estimator keeps its last successful image as the reference and retries with a
wider baseline. The plot therefore shows gaps for failed frames.

## Setup

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Place the KITTI odometry dataset under `data/kitti/`. This example uses the
grayscale odometry images and the separately distributed ground-truth poses.
Ground-truth pose files should be at
`data/kitti/data_odometry_poses/dataset/poses/<sequence>.txt`; poses are
available for sequences `00` through `10`.

Run the example from the repository root:

```bash
python src/main.py
```

The example runs sequence `02` using SIFT features and plots the estimated
trajectory alongside ground truth. The dataset loader reads and checks both
camera image sequences, timestamps, and calibration, but the current odometry
algorithm uses only the left-camera (`image_0`) frames.

## Modules

- `src/main.py` — example entry point and run configuration.
- `src/dataset.py` — KITTI image sequences, timestamps, calibration, and
  optional ground-truth poses.
- `src/camera.py` — camera-coordinate and calibration notes.
- `src/features.py` — feature detection, descriptors, and matching.
- `src/motion.py` — essential-matrix estimation and relative-pose recovery.
- `src/odometry.py` — monocular motion estimation and trajectory accumulation.
- `src/visualize.py` — trajectory plotting.

The example uses SIFT. Change `feature_type="sift"` to `feature_type="orb"` in
`src/main.py` to use ORB instead. Descriptor matching selects L2 distance for
SIFT and Hamming distance for ORB.

The trajectory plot is a visual comparison only. It does not compute ATE or
RPE metrics or align monocular scale to ground truth.
