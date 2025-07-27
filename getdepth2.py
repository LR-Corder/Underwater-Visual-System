# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import numpy as np
import camera_configs

# --------------------------------------------------------------------------- #
# Stereo depth utilities
# --------------------------------------------------------------------------- #
_BLOCK_SIZE = 13
_PARAMS = {
    "minDisparity": 0,
    "numDisparities": 16 * 4,
    "blockSize": _BLOCK_SIZE,
    "P1": 8 * 3 * _BLOCK_SIZE * _BLOCK_SIZE,
    "P2": 32 * 3 * _BLOCK_SIZE * _BLOCK_SIZE,
    "disp12MaxDiff": 20,
    "preFilterCap": 15,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 2,
    "mode": cv2.STEREO_SGBM_MODE_HH,
}

_LEFT_MATCHER = cv2.StereoSGBM_create(**_PARAMS)
_RIGHT_MATCHER = cv2.ximgproc.createRightMatcher(_LEFT_MATCHER)

_WLS = cv2.ximgproc.createDisparityWLSFilter(_LEFT_MATCHER)
_WLS.setLambda(8000.0)
_WLS.setSigmaColor(1.3)
_WLS.setLRCthresh(24)
_WLS.setDepthDiscontinuityRadius(3)


def _rectify(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undistort and rectify stereo pair."""
    left_r = cv2.remap(left, camera_configs.left_map1, camera_configs.left_map2,
                       cv2.INTER_LINEAR)
    right_r = cv2.remap(right, camera_configs.right_map1, camera_configs.right_map2,
                        cv2.INTER_LINEAR)
    return left_r, right_r


def _compute_depth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return full 3-D point cloud (H×W×3)."""
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp = _LEFT_MATCHER.compute(left_gray, right_gray)
    right_disp = _RIGHT_MATCHER.compute(right_gray, left_gray)

    filtered = _WLS.filter(left_disp, left_gray,
                           disparity_map_right=right_disp)
    return cv2.reprojectImageTo3D(filtered.astype(np.float32) / 16.0,
                                  camera_configs.Q)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def get_depth(frame1: np.ndarray, frame2: np.ndarray, x: float, y: float) -> np.ndarray:
    """Return 3-D point cloud from stereo pair."""
    left, right = _rectify(frame1, frame2)
    return _compute_depth(left, right)


def get_depthmap(frame1: np.ndarray, frame2: np.ndarray, x: float, y: float) -> np.ndarray:
    """Return 8-bit disparity map for display."""
    left, right = _rectify(frame1, frame2)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp = _LEFT_MATCHER.compute(left_gray, right_gray)
    right_disp = _RIGHT_MATCHER.compute(right_gray, left_gray)
    filtered = _WLS.filter(left_disp, left_gray,
                           disparity_map_right=right_disp)
    return cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)