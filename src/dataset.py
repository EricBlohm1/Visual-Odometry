from pathlib import Path
import numpy as np
from PIL import Image


class KittiOdometryDataset:
    def __init__(self, data_path: str, sequence: str, poses_dir: str | Path | None = None):
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


        self.right_images = self.sequence_path / "image_1"
        self.right_images = sorted(
            self.right_images.glob("*.png"),
            key=lambda x: int(x.stem)  # Stem removes 0 padding
        )
        if not self.right_images:
            raise FileNotFoundError(f"No right images found in {self.right_images}.")



        # ------ Timestamps -------
        times_path = self.sequence_path / "times.txt"
        self.timestamps = np.loadtxt(times_path)

        if len(self.timestamps) != len(self.left_images):
            raise ValueError(
                f"Number of timestamps ({len(self.timestamps)}) does not match "
                f"number of left images ({len(self.left_images)})."
            )

        if len(self.timestamps) != len(self.right_images):
            raise ValueError(
                f"Number of timestamps ({len(self.timestamps)}) does not match "
                f"number of right images ({len(self.right_images)})."
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
        self.poses = None
        poses_path = (
            Path(poses_dir).expanduser() / f"{self.sequence}.txt"
            if poses_dir is not None
            else self.sequence_path / "poses.txt"
        )
        if poses_path.is_file():
            raw_poses = np.loadtxt(poses_path, ndmin=2)
            if raw_poses.shape != (len(self.left_images), 12):
                raise ValueError(
                    f"Expected {len(self.left_images)} pose rows with 12 values "
                    f"each, but found shape {raw_poses.shape} in {poses_path}."
                )

            self.poses = np.repeat(
                np.eye(4)[None, :, :], len(raw_poses), axis=0
            )
            self.poses[:, :3, :4] = raw_poses.reshape(-1, 3, 4)
        elif poses_dir is not None:
            raise FileNotFoundError(f"Ground-truth poses not found: {poses_path}")


    def __len__(self):
         return len(self.left_images)

    def get_cam0(self, index: int):
        with Image.open(self.left_images[index]) as img:
            return np.array(img)

    def get_cam1(self, index: int):
        with Image.open(self.right_images[index]) as img:
            return np.array(img)
