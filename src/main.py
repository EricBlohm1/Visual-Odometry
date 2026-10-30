from dataset import KittiOdometryDataset
from odometry import VisualOdometry
from visualize import plot_trajectory

if __name__ == "__main__":
    dataset = KittiOdometryDataset(
        "./data/kitti/data_odometry_gray/dataset",
        "00",
        poses_dir="./data/kitti/data_odometry_poses/dataset/poses",
    )
    K = dataset.P0[:3, :3]
    ground_truth = dataset.poses

    odometry = VisualOdometry(K, feature_type="sift")
    trajectory = odometry.run(dataset)
    plot_trajectory(trajectory, ground_truth)

