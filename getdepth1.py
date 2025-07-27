# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import time
import numpy as np
import cv2

import camera_configs

# --------------------------------------------------------------------------- #
# Stereo depth estimation (SGBM + WLS)
# --------------------------------------------------------------------------- #
def _build_stereo_matcher(block_size: int = 5, mode: int = 2) -> tuple:
    """Instantiate SGBM and WLS objects."""
    params = {
        "minDisparity": 0,
        "numDisparities": 16 * 4,
        "blockSize": block_size,
        "P1": 8 * 3 * block_size * block_size,
        "P2": 32 * 3 * block_size * block_size,
        "disp12MaxDiff": 0,
        "preFilterCap": 15,
        "uniquenessRatio": 5,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": mode,
    }
    left_matcher = cv2.StereoSGBM_create(**params)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    wls = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.3)
    wls.setLRCthresh(24)
    wls.setDepthDiscontinuityRadius(3)
    return left_matcher, right_matcher, wls


LEFT_MATCHER, RIGHT_MATCHER, WLS_FILTER = _build_stereo_matcher()


def _compute_depth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return 3-D point cloud (H×W×3) from rectified stereo pair."""
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp = LEFT_MATCHER.compute(left_gray, right_gray)
    right_disp = RIGHT_MATCHER.compute(right_gray, left_gray)

    filtered = WLS_FILTER.filter(left_disp, left_gray,
                                 disparity_map_right=right_disp)
    return cv2.reprojectImageTo3D(filtered.astype(np.float32) / 16.0,
                                  camera_configs.Q)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_depth_(frame1: np.ndarray, frame2: np.ndarray, x: float, y: float) -> np.ndarray:
    """Return the full 3-D point cloud."""
    left = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2,
                     cv2.INTER_LINEAR)
    right = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2,
                      cv2.INTER_LINEAR)
    return _compute_depth(left, right)


def get_depthmap(frame1: np.ndarray, frame2: np.ndarray, x: float, y: float) -> np.ndarray:
    """Return disparity map as 8-bit image for display."""
    left = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2,
                     cv2.INTER_LINEAR)
    right = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2,
                      cv2.INTER_LINEAR)

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp = LEFT_MATCHER.compute(left_gray, right_gray)
    right_disp = RIGHT_MATCHER.compute(right_gray, left_gray)
    filtered = WLS_FILTER.filter(left_disp, left_gray,
                                 disparity_map_right=right_disp)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


def get_depth(frame1: np.ndarray, frame2: np.ndarray, x: float, y: float,
              block_size: int = 5) -> np.ndarray:
    """Legacy alias returning 3-D point cloud with configurable block size."""
    matcher, right_matcher, wls = _build_stereo_matcher(block_size)
    left = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2,
                     cv2.INTER_LINEAR)
    right = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2,
                      cv2.INTER_LINEAR)

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp = matcher.compute(left_gray, right_gray)
    right_disp = right_matcher.compute(right_gray, left_gray)
    filtered = wls.filter(left_disp, left_gray, disparity_map_right=right_disp)
    return cv2.reprojectImageTo3D(filtered.astype(np.float32) / 16.0,
                                  camera_configs.Q)