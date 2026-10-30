import cv2
import numpy as np
from features import (
    create_detector,
    get_features,
    match_features,
    visualize_matches as draw_feature_matches,
)
from motion import estimate_essential_matrix, recover_pose


class VisualOdometry:
    def __init__(self, K, feature_type="orb", nfeatures=2000):
        self.K = np.asarray(K, dtype=np.float64)
        if self.K.shape != (3, 3) or not np.isfinite(self.K).all():
            raise ValueError("K must be a finite 3x3 camera intrinsic matrix.")
        self.feature_type = feature_type.lower()
        self.detector = create_detector(self.feature_type, nfeatures)
        self.T_world = np.eye(4)
        self.trajectory = [self.T_world.copy()]

    def run(self, dataset, visualize_matches=False):
        if len(dataset) == 0:
            raise ValueError("Cannot run visual odometry on an empty dataset.")

        self.T_world = np.eye(4)
        self.trajectory = [self.T_world.copy()]
        prev_img = dataset.get_cam0(0)
        prev_kp, prev_des = get_features(prev_img, self.detector)

        for k in range(1, len(dataset)):
            curr_img = dataset.get_cam0(k)
            curr_kp, curr_des = get_features(curr_img, self.detector)
            matches = match_features(prev_des, curr_des)
            essential, points_prev, points_curr, ransac_mask = (
                estimate_essential_matrix(prev_kp, curr_kp, matches, self.K)
            )

            if visualize_matches:
                visualization = draw_feature_matches(
                    prev_img, prev_kp, curr_img, curr_kp, matches
                )
                cv2.imshow("Feature matches", visualization)
                cv2.waitKey(1)

            if essential is None:
                self.trajectory.append(np.full((4, 4), np.nan))
                continue

            R, t, _ = recover_pose(
                essential, points_prev, points_curr, self.K, ransac_mask
            )
            # Do not accumulate pose if the motion estimation failed
            if R is None:
                self.trajectory.append(np.full((4, 4), np.nan))
                continue

            T_prev_curr = np.eye(4)
            T_prev_curr[:3, :3] = R.T
            T_prev_curr[:3, 3] = (-R.T @ t).reshape(3)
            self.T_world = self.T_world @ T_prev_curr

            self.trajectory.append(self.T_world.copy())
            prev_img = curr_img
            prev_kp, prev_des = curr_kp, curr_des

        return np.array(self.trajectory)
