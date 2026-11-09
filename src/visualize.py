import matplotlib.pyplot as plt
import numpy as np


def _trajectory_positions(trajectory):
    positions = np.asarray(trajectory)[:, :3, 3].copy()
    positions -= positions[0]
    return positions


class LiveTrajectoryPlot:
    def __init__(self, ground_truth=None):
        plt.ion()
        self.figure, self.ax = plt.subplots()
        (self.estimated_line,) = self.ax.plot([], [], label="Estimated trajectory")
        self.start_marker = self.ax.scatter([], [], label="Start")
        self.last_marker = self.ax.scatter([], [], label="Last valid pose")

        if ground_truth is not None:
            ground_truth_positions = _trajectory_positions(ground_truth)
            self.ax.plot(
                ground_truth_positions[:, 0],
                ground_truth_positions[:, 2],
                "--",
                label="Ground truth",
            )

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Z (m)")
        self.ax.set_title("Estimated trajectory vs. ground truth")
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.grid(True)
        self.ax.legend()
        plt.show(block=False)
        self.update(np.eye(4)[None, :, :])

    def update(self, trajectory):
        positions = _trajectory_positions(trajectory)
        valid_positions = positions[np.isfinite(positions).all(axis=1)]

        self.estimated_line.set_data(positions[:, 0], positions[:, 2])
        if len(valid_positions):
            self.start_marker.set_offsets(valid_positions[0, [0, 2]][None, :])
            self.last_marker.set_offsets(valid_positions[-1, [0, 2]][None, :])
        else:
            self.start_marker.set_offsets(np.empty((0, 2)))
            self.last_marker.set_offsets(np.empty((0, 2)))

        self.ax.relim()
        self.ax.autoscale_view()
        self.figure.canvas.draw_idle()
        plt.pause(0.001)

    def show(self):
        plt.ioff()
        plt.show()


def plot_trajectory(trajectory, ground_truth=None):
    # Each pose is a 4x4 camera-to-world transform.
    positions = _trajectory_positions(trajectory)
    ground_truth_positions = None
    estimated_label = "Estimated trajectory"
    if ground_truth is not None:
        ground_truth_positions = _trajectory_positions(ground_truth)
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

    axis_scale = "m"
    ax.set_xlabel(f"X ({axis_scale})")
    ax.set_ylabel(f"Z ({axis_scale})")
    ax.set_title("Estimated trajectory vs. ground truth")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True)
    ax.legend()
    plt.show()