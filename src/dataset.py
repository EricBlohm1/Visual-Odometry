from pathlib import Path
import numpy as np
from PIL import Image


# Will only use left camera for visual odometry, but can add the one for stereo later.
class KittiOdometryDataset:
    def __init__(self, data_path: str, sequence: str):
        self.root = str(Path(data_path).expanduser())
        self.sequence = sequence
        self.sequence_path = Path(self.root) / "sequences" / self.sequence

        if not self.sequence_path.is_dir():
            raise ValueError(f"Sequence path {self.sequence_path} not found.")


        # ------- Images -------
        self.left_images = self.sequence_path / "image_0" 
        self.left_images = sorted(
            self.left_images.glob("*.png"), 
            key=lambda x: int(x.stem) # Stem removes 0 padding
        )
        if not self.left_images:
            raise FileNotFoundError(f"No left images found in {self.left_images}.")


        # ------ Timestamps -------
        times_path = self.sequence_path / "times.txt"
        self.timestamps = np.loadtxt(times_path)

        if len(self.timestamps) != len(self.left_images):
            raise ValueError(
                f"Number of timestamps ({len(self.timestamps)}) does not match "
                f"number of left images ({len(self.left_images)})."
            )



        #------ Calibration -------
        calib_path = self.sequence_path / "calib.txt"
        self.calibration = {}
        with open(calib_path, "r") as f:
            for line in f:
                key, value = line.split(":", 1)
                self.calibration[key] = np.fromstring(value, sep=" ")

        self.P0 = self.calibration.get('P0').reshape(3,4)
        self.P1 = self.calibration.get('P1').reshape(3,4)


        #------ Optional Ground Truth Poses -------
        poses_path = self.sequence_path / "poses.txt"
        self.poses = None

        if poses_path.is_file():
            raw_poses = np.loadtxt(poses_path)
            self.poses = raw_poses.reshape(-1,3,4) # Each pose i a 3x4 matrix.


    def __len__(self):
         return len(self.left_images)

    def get_cam0(self, index: int):
        with Image.open(self.left_images[index]) as img:
            return np.array(img)

