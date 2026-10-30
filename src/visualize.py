import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(trajectory, ground_truth=None):
    # Each pose is a 4x4 camera-to-world transform.
    positions = np.asarray(trajectory)[:, :3, 3].copy()
    ground_truth_positions = None
    estimated_label = "Estimated (monocular scale)"
    if ground_truth is not None:
        ground_truth_positions = np.asarray(ground_truth)[:, :3, 3].copy()
        positions -= positions[0]
        ground_truth_positions -= ground_truth_positions[0]
    else:
        positions -= positions[0]
    valid_positions = positions[np.isfinite(positions).all(axis=1)]

    _, ax = plt.subplots()
    ax.plot(
        positions[:, 0],
        positions[:, 2],
        label=estimated_label,
    )

    if ground_truth_positions is not None:
        ax.plot(
            ground_truth_positions[:, 0],
            ground_truth_positions[:, 2],
            "--",
            label="Ground truth",
        )

    if len(valid_positions):
        ax.scatter(valid_positions[0, 0], valid_positions[0, 2], label="Start")
        ax.scatter(valid_positions[-1, 0], valid_positions[-1, 2], label="Last valid pose")

    axis_scale = "arbitrary scale"
    ax.set_xlabel(f"X ({axis_scale})")
    ax.set_ylabel(f"Z ({axis_scale})")
    ax.set_title("Estimated trajectory vs. ground truth")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()
    plt.show()