import argparse
from pathlib import Path

import numpy as np

from dataset import KittiOdometryDataset
from odometry import VisualOdometry
from visualize import LiveTrajectoryPlot


def create_parser():
    parser = argparse.ArgumentParser(
        description="Run stereo visual odometry on a KITTI odometry sequence."
    )
    parser.add_argument(
        "--sequence",
        default="00",
        help="KITTI sequence number (default: 00).",
    )
    parser.add_argument(
        "--features",
        choices=("sift", "orb"),
        default="sift",
        help="Feature detector to use (default: sift).",
    )
    return parser


def main():
    args = create_parser().parse_args()
    dataset_dir = Path("data/kitti/data_odometry_gray/dataset")
    poses_dir = Path("data/kitti/data_odometry_poses/dataset/poses")
    dataset = KittiOdometryDataset(
        str(dataset_dir),
        args.sequence,
        poses_dir=poses_dir,
    )
    if dataset.poses is None:
        raise ValueError(f"Ground-truth poses are required for sequence {args.sequence}.")

    odometry = VisualOdometry(dataset.P0[:3, :3], feature_type=args.features)
    live_plot = LiveTrajectoryPlot(dataset.poses)
    trajectory = odometry.run(
        dataset,
        trajectory_callback=live_plot.update,
        update_every=5,
    )

    if trajectory.shape != dataset.poses.shape:
        raise ValueError(
            f"Expected one estimated pose per frame ({len(dataset)}), "
            f"but received {len(trajectory)} poses."
        )
    if not np.isfinite(trajectory).all() or not np.isfinite(dataset.poses).all():
        raise ValueError(
            "Cannot save trajectories containing non-finite poses; "
            "estimate and ground truth must remain frame-aligned."
        )

    output_dir = Path("results/estimated_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)

    estimate_path = output_dir / f"{args.sequence}.txt"
    ground_truth_path = output_dir / f"{args.sequence}_gt.txt"

    np.savetxt(estimate_path, trajectory[:, :3, :].reshape(-1, 12), fmt="%.9f")
    np.savetxt(
        ground_truth_path,
        dataset.poses[:, :3, :].reshape(-1, 12),
        fmt="%.9f",
    )

    print(f"Saved estimate to {estimate_path}")
    print(f"Saved matching ground truth to {ground_truth_path}")
    live_plot.show()


if __name__ == "__main__":
    main()
