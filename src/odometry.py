import cv2
import numpy as np

from camera import (
    build_pnp_correspondences,
    get_stereo_3d,
)
from features import (
    create_detector,
    get_features,
    match_features,
    visualize_matches as draw_feature_matches,
)


class VisualOdometry:
    def __init__(self, K, feature_type="sift", nfeatures=3000, min_pnp_inliers=10):
        self.K = np.asarray(K)
        self.feature_type = feature_type.lower()
        self.detector = create_detector(self.feature_type, nfeatures)
        self.T_world = np.eye(4)
        self.trajectory = [self.T_world.copy()]
        self.object_points = np.empty((0, 3))
        self.image_points = np.empty((0, 2))
        self.min_pnp_inliers = min_pnp_inliers

    def estimate_motion(self):
        if (
            len(self.object_points) < 4
            or len(self.image_points) < 4 # otherwise ambiguous solution
        ):
            return None

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            self.object_points,
            self.image_points,
            self.K,
            None,
            iterationsCount=300,
            reprojectionError=2,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if (
            not success
            or inliers is None
            or len(inliers) < self.min_pnp_inliers
        ):
            return None

        inlier_indices = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(
            self.object_points[inlier_indices],
            self.image_points[inlier_indices],
            self.K,
            None,
            rvec,
            tvec,
        )

        rotation, _ = cv2.Rodrigues(rvec)
        relative_transform = np.eye(4)
        relative_transform[:3, :3] = rotation.T
        relative_transform[:3, 3] = -rotation.T @ tvec.reshape(3)
        return relative_transform

    def run(
        self,
        dataset,
        trajectory_callback,
        update_every,
    ):
        P0 = np.asarray(dataset.P0, dtype=np.float64)
        P1 = np.asarray(dataset.P1, dtype=np.float64)

        self.T_world = np.eye(4)
        self.trajectory = [self.T_world.copy()]
        last_relative_transform = np.eye(4)


        # Initialize reference frame
        reference_left = dataset.get_cam0(0)
        reference_right = dataset.get_cam1(0)
        reference_kp_left, reference_des_left = get_features(reference_left, self.detector)
        reference_kp_right, reference_des_right = get_features(reference_right, self.detector)

        reference_stereo_matches = match_features(
            reference_des_left, reference_des_right
        )

        reference_points_3d, reference_left_indices = get_stereo_3d(
            reference_kp_left,
            reference_kp_right,
            reference_stereo_matches,
            self.K,
            P0,
            P1,
            min_depth=3.0,
            max_depth=60.0,
            max_vertical_error=1.0,
        )
        # --------------------------------

        for frame_index in range(1, len(dataset)):
            if frame_index % 10 == 0:
                print(f"Processing frame {frame_index}/{len(dataset)}")
            current_left = dataset.get_cam0(frame_index)
            current_right = dataset.get_cam1(frame_index)
            current_kp_left, current_des_left = get_features(current_left, self.detector)
            current_kp_right, current_des_right = get_features(current_right, self.detector)

            current_stereo_matches = match_features(
                current_des_left, current_des_right
            )

            current_points_3d, current_left_indices = get_stereo_3d(
                current_kp_left,
                current_kp_right,
                current_stereo_matches,
                self.K,
                P0,
                P1,
                min_depth=3.0,
                max_depth=60.0,
                max_vertical_error=1.0,
            )


            temporal_matches = match_features(reference_des_left, current_des_left)
            object_points, image_points = build_pnp_correspondences(
                reference_points_3d,
                reference_left_indices,
                temporal_matches,
                current_kp_left,
            )
            self.object_points = object_points
            self.image_points = image_points

            relative_transform = self.estimate_motion()
            if relative_transform is None:
                print(
                    f"PnP failed or fewer than {self.min_pnp_inliers} inliers "
                    f"at frame {frame_index}"
                )
                self.T_world = self.T_world @ last_relative_transform
                self.trajectory.append(self.T_world.copy())
            else:
                self.T_world = self.T_world @ relative_transform
                last_relative_transform = relative_transform.copy()
                self.trajectory.append(self.T_world.copy())

            reference_left = current_left
            reference_right = current_right
            reference_kp_left = current_kp_left
            reference_des_left = current_des_left
            reference_kp_right = current_kp_right
            reference_des_right = current_des_right
            reference_points_3d = current_points_3d
            reference_left_indices = current_left_indices
            reference_frame_idx = frame_index

            if trajectory_callback is not None and (
                frame_index % update_every == 0
                or frame_index == len(dataset) - 1
            ):
                trajectory_callback(np.asarray(self.trajectory))

        return np.asarray(self.trajectory)
