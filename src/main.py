# TODO 1: Inspect the KITTI directory structure.
# TODO 2: Load and inspect a single image.
# TODO 3: Load a sequence of images in the correct order.
# TODO 4: Read camera calibration and understand coordinate conventions.
# TODO 5: Detect features in two consecutive frames.
# TODO 6: Match features and inspect the correspondences.
# TODO 7: Estimate relative camera motion.
# TODO 8: Accumulate poses into a trajectory.
# TODO 9: Visualize and evaluate the trajectory.
#
# Tips:
# - Validate each stage independently and start with a small number of frames.
# - Inspect intermediate results visually.
# - Check array shapes, matrix dimensions, coordinate frames, and units.
# - Separate data-loading errors from geometric or algorithmic errors.
# - Study the relevant mathematics before implementing each stage.


from dataset import KittiOdometryDataset
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    dataset = KittiOdometryDataset("./data/kitti/data_odometry_gray/dataset", "00")
    img = dataset.get_cam0(0)  # Get the first left camera image

    # Initial camera pose (world frame)
    T_world = np.eye(4)
    trajectory = [T_world.copy()]

    # Load the first frame
    prev_img = dataset.get_cam0(0)

    for k in range(1, len(dataset)):
        curr_img = dataset.get_cam0(k)

        # 1. Detect and match features between prev_img and curr_img
        # 2. Estimate the essential matrix E using matched points
        # 3. Recover relative rotation R and translation direction t
        # 4. Construct the relative transformation T_curr_prev
        # 5. Update T_world using the correct transform convention

        # Placeholder: replace with your pose update
        # T_world = ...

        trajectory.append(T_world.copy())

        # Advance to the next frame pair
        prev_img = curr_img

    trajectory = np.array(trajectory)
    print("Trajectory shape:", trajectory.shape)