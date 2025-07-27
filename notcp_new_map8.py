# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import numpy as np
import time
from get_o_point import result_point
import camera_configs

# --------------------------------------------------------------------------- #
# Stereo parameters (from MATLAB calibration)
# --------------------------------------------------------------------------- #
LEFT_INTRINSIC = np.array([[630.6456, 0.2942, 342.7966],
                           [0, 630.6821, 292.2809],
                           [0, 0, 1]])
RIGHT_INTRINSIC = np.array([[632.1360, 0.4287, 356.9454],
                            [0, 632.1411, 290.7813],
                            [0, 0, 1]])
LEFT_DISTORTION = np.array([0.4072, 0.0966, -0.0013, -0.0054, -0.1140])
RIGHT_DISTORTION = np.array([0.3920, 0.1952, -0.0011, 0.0022, -0.4103])

R = np.array([[0.9999, 0.0015, -0.0170],
              [-0.0015, 1.0000, -0.0001],
              [0.0170, 0.0001, 0.9999]])
T = np.array([170.5653, 0.3794, 2.6282])

SIZE = (1280, 480)
R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(LEFT_INTRINSIC, LEFT_DISTORTION,
                                          RIGHT_INTRINSIC, RIGHT_DISTORTION,
                                          SIZE, R, T)
LEFT_MAP1, LEFT_MAP2 = cv2.initUndistortRectifyMap(LEFT_INTRINSIC, LEFT_DISTORTION,
                                                   R1, P1, SIZE, cv2.CV_16SC2)
RIGHT_MAP1, RIGHT_MAP2 = cv2.initUndistortRectifyMap(RIGHT_INTRINSIC, RIGHT_DISTORTION,
                                                     R2, P2, SIZE, cv2.CV_16SC2)

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def _fill_inf(depth: np.ndarray) -> np.ndarray:
    """Replace inf values with linear interpolation along columns."""
    for c in range(depth.shape[2]):
        mask = np.isinf(depth[:, :, c])
        valid = ~mask
        depth[mask, c] = np.interp(np.where(mask)[1],
                                   np.where(valid)[1],
                                   depth[valid, c])
    return depth


def _rectify_pair(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap stereo images to rectified coordinates."""
    left_r = cv2.remap(left, LEFT_MAP1, LEFT_MAP2, cv2.INTER_LINEAR)
    right_r = cv2.remap(right, RIGHT_MAP1, RIGHT_MAP2, cv2.INTER_LINEAR)
    return left_r, right_r


def _compute_depth(left_gray: np.ndarray, right_gray: np.ndarray) -> np.ndarray:
    """Compute 3-D point cloud using SGBM + WLS."""
    matcher = cv2.StereoSGBM_create(minDisparity=1,
                                    numDisparities=64,
                                    blockSize=4,
                                    P1=8 * 3 * 4 * 4,
                                    P2=32 * 3 * 4 * 4,
                                    disp12MaxDiff=-1,
                                    preFilterCap=1,
                                    uniquenessRatio=10,
                                    speckleWindowSize=100,
                                    speckleRange=100,
                                    mode=cv2.STEREO_SGBM_MODE_HH)

    right_matcher = cv2.ximgproc.createRightMatcher(matcher)
    wls = cv2.ximgproc.createDisparityWLSFilter(matcher)
    wls.setLambda(4000)
    wls.setSigmaColor(2)
    wls.setLRCthresh(30)
    wls.setDepthDiscontinuityRadius(5)

    left_disp = matcher.compute(left_gray, right_gray)
    right_disp = right_matcher.compute(right_gray, left_gray)
    filtered = wls.filter(left_disp, left_gray, disparity_map_right=right_disp)

    disp_norm = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    threeD = cv2.reprojectImageTo3D(filtered, Q, handleMissingValues=True)
    return threeD * 16 / 1000, disp_norm  # mm → m


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    left_raw = frame[:, 640:]
    right_raw = frame[:, :640]

    left, right = _rectify_pair(left_raw, right_raw)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    threeD, disp_vis = _compute_depth(left_gray, right_gray)
    threeD = _fill_inf(threeD)

    disp_bgr = cv2.cvtColor(disp_vis, cv2.COLOR_GRAY2BGR)
    targets = result_point(disp_bgr, threeD)

    for x, y, _, (w, h) in targets:
        z = threeD[y, x, 2]
        print(f"x:{x:.1f}, y:{y:.1f}, z:{z:.2f}, w:{w:.1f}, h:{h:.1f}")

    cv2.imshow('left', left)
    cv2.imshow('right', right)
    cv2.imshow('opencv_norm', disp_vis)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()