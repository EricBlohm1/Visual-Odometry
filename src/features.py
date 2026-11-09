import cv2
import numpy as np


def create_detector(feature_type, nfeatures):
    feature_type = feature_type.lower()
    if feature_type == "orb":
        return cv2.ORB_create(nfeatures=nfeatures)
    if feature_type == "sift":
        return cv2.SIFT_create(nfeatures=nfeatures)
    
    raise ValueError(
        f"Unsupported feature type {feature_type!r}; choose 'orb' or 'sift'."
    )


def get_features(img, detector):
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors


def match_features(des1, des2):
    if des1 is None or des2 is None:
        return []

    if len(des1) < 2 or len(des2) < 2:
        return []

    if des1.dtype != des2.dtype:
        raise ValueError("Descriptor arrays must have matching data types.")

    # Depends on the feature type: ORB uses Hamming distance, SIFT uses L2 distance.
    norm = cv2.NORM_HAMMING if des1.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    candidates = matcher.knnMatch(des1, des2, k=2)

    good_matches = []
    for pair in candidates:
        if len(pair) < 2:
            continue
        best, second_best = pair

        #Lowe ratio test to filter out ambiguous matches
        if best.distance < 0.75 * second_best.distance:
            good_matches.append(best)
    return good_matches


def visualize_matches(img1, kp1, img2, kp2, matches, max_matches=10):
    matches_viz = sorted(
        matches,
        key=lambda m: m.distance
    )[:max_matches]

    visualization = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches_viz, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return visualization