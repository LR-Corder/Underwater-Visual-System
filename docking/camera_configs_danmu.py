# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import numpy as np

# Camera intrinsics and distortion (mono camera)
LEFT_INTRINSIC = np.array([
    [377.4276236643150, 0.0031288096633, 319.6520423980789],
    [0, 377.6352854802945, 218.2777212015161],
    [0, 0, 1]
])
LEFT_DISTORTION = np.array([
    -0.001920627940025, -0.015728408450596, -0.000372018458586,
    -0.000157976298467, 0.003219191673555
])

RIGHT_INTRINSIC = np.array([
    [465.4012, -0.8540, 338.8338],
    [0, 465.3984, 298.6513],
    [0, 0, 1]
])
RIGHT_DISTORTION = np.array([
    0.010200, -0.077743, -0.000413, -0.000563, 0.032876
])

# Stereo extrinsics
R = np.array([
    [0.9999, -0.0123, -0.0059],
    [0.0124, 0.9998, 0.0151],
    [0.0057, -0.0152, 0.9999]
])
T = np.array([[-171.3975], [-0.4235], [1.9587]])

SIZE = (640, 480)

# Rectification maps
R1, R2, P1, P2, Q, *_ = cv2.stereoRectify(
    LEFT_INTRINSIC, LEFT_DISTORTION,
    RIGHT_INTRINSIC, RIGHT_DISTORTION,
    SIZE, R, T
)

LEFT_MAP1, LEFT_MAP2 = cv2.initUndistortRectifyMap(
    LEFT_INTRINSIC, LEFT_DISTORTION, R1, P1, SIZE, cv2.CV_16SC2
)
RIGHT_MAP1, RIGHT_MAP2 = cv2.initUndistortRectifyMap(
    RIGHT_INTRINSIC, RIGHT_DISTORTION, R2, P2, SIZE, cv2.CV_16SC2
)