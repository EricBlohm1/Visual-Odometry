import cv2
import numpy as np


def get_stereo_3d(
    kp_left,
    kp_right,
    matches,
    K0,
    P0,
    P1,
    min_depth,
    max_depth,
    max_vertical_error,
):
    if not np.isfinite([min_depth, max_depth]).all():
        raise ValueError("Depth limits must be finite.")
    if min_depth <= 0 or max_depth <= min_depth:
        raise ValueError("Require 0 < min_depth < max_depth.")
    if not np.isfinite(max_vertical_error) or max_vertical_error < 0:
        raise ValueError("max_vertical_error must be finite and non-negative.")

    fx = K0[0, 0]
    fy = K0[1, 1]

    camera0_x = -P0[0, 3] / P0[0, 0]
    camera1_x = -P1[0, 3] / P1[0, 0]
    baseline = camera1_x - camera0_x
    if baseline <= 0:
        raise ValueError("P0/P1 must describe a positive left-to-right baseline.")

    cx_left = K0[0, 2]
    cy_left = K0[1, 2]
    cx_right = P1[0, 2] / P1[2, 2]

    left_indices = []
    points_3d = []
    for match in matches:

        u_left, v_left = kp_left[match.queryIdx].pt
        u_right, v_right = kp_right[match.trainIdx].pt
        if abs(v_left - v_right) > max_vertical_error:
            continue

        disparity = (u_left - cx_left) - (u_right - cx_right)
        if not np.isfinite(disparity) or disparity <= 0:
            continue

        depth = fx * baseline / disparity
        if not np.isfinite(depth):
            continue
        if (
            depth < min_depth
            and not np.isclose(depth, min_depth)
        ) or (
            depth > max_depth
            and not np.isclose(depth, max_depth)
        ):
            continue

        x = (u_left - cx_left) * depth / fx
        y = (v_left - cy_left) * depth / fy
        if not np.isfinite([x, y, depth]).all():
            continue

        points_3d.append((x, y, depth))
        left_indices.append(match.queryIdx)

    return np.asarray(points_3d, dtype=np.float32).reshape(-1, 3), left_indices


def build_pnp_correspondences(
    reference_points_3d,
    reference_left_indices,
    temporal_matches,
    current_kp_left
):
    # Map reference-left keypoint indices to their 3D points
    points_3d_map = dict(
        zip(reference_left_indices, reference_points_3d)
    )

    object_points = []
    image_points = []

    for match in temporal_matches:

        # Index in reference-left image
        reference_idx = match.queryIdx

        # Index in current-left image
        current_idx = match.trainIdx

        # Check whether the reference feature has a valid 3D point
        if reference_idx not in points_3d_map:
            continue

        # Retrieve corresponding 3D and 2D observations
        point_3d = points_3d_map[reference_idx]
        point_2d = current_kp_left[current_idx].pt

        object_points.append(point_3d)
        image_points.append(point_2d)

    return (
        np.asarray(object_points, dtype=np.float32),
        np.asarray(image_points, dtype=np.float32)
    )
