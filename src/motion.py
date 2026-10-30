import cv2
import numpy as np


def estimate_essential_matrix(kp1, kp2, matches, K):
    K = np.asarray(K, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("K must be a finite 3x3 camera intrinsic matrix.")

    if len(matches) < 5:
        return None, None, None, None

    pts1 = np.float64([
        kp1[m.queryIdx].pt for m in matches
    ]).reshape(-1,1,2)

    pts2 = np.float64([
        kp2[m.trainIdx].pt for m in matches
    ]).reshape(-1, 1, 2)

    if not np.isfinite(pts1).all() or not np.isfinite(pts2).all():
        raise ValueError("Matched image points must be finite.")

    E, mask = cv2.findEssentialMat(
        pts1,
        pts2,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    return E, pts1, pts2, mask


def recover_pose(E, pts1, pts2, K, mask=None):
    if E is None:
        return None, None, None

    E = np.asarray(E, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64).reshape(-1, 1, 2)
    pts2 = np.asarray(pts2, dtype=np.float64).reshape(-1, 1, 2)
    K = np.asarray(K, dtype=np.float64)

    inlier_count, R, t, pose_mask = cv2.recoverPose(
        E, pts1, pts2, K, mask=mask
    )

    if (inlier_count < 5):
        return None, None, pose_mask

    return R, t, pose_mask
