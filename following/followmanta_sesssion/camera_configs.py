# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.
import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Calibration data for 2024-09-20 air test – wide-angle, upright
# --------------------------------------------------------------------------- #
LEFT_INTRINSIC = np.array([
    [337.7759, -0.3629, 359.3753],
    [0.0, 337.2240, 294.3447],
    [0.0, 0.0, 1.0],
])
LEFT_DISTORTION = np.array([-0.321095, 0.123865, 0.002685, 0.000386, -0.023545])

RIGHT_INTRINSIC = np.array([
    [339.5218, 0.6127, 341.7623],
    [0.0, 340.5494, 283.3478],
    [0.0, 0.0, 1.0],
])
RIGHT_DISTORTION = np.array([-0.315224, 0.112723, -0.000791, -0.000649, -0.019773])

R = np.array([
    [1.0000, 0.0053, 0.0077],
    [-0.0054, 0.9999, 0.0099],
    [-0.0077, -0.0099, 0.9999],
])
T = np.array([-99.9209, -5.9070, 2.8822])

SIZE = (640, 480)

# --------------------------------------------------------------------------- #
# Rectification maps
# --------------------------------------------------------------------------- #
R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(
    LEFT_INTRINSIC, LEFT_DISTORTION,
    RIGHT_INTRINSIC, RIGHT_DISTORTION,
    SIZE, R, T,
)

LEFT_MAP1, LEFT_MAP2 = cv2.initUndistortRectifyMap(
    LEFT_INTRINSIC, LEFT_DISTORTION, R1, P1, SIZE, cv2.CV_16SC2
)
RIGHT_MAP1, RIGHT_MAP2 = cv2.initUndistortRectifyMap(
    RIGHT_INTRINSIC, RIGHT_DISTORTION, R2, P2, SIZE, cv2.CV_16SC2
)