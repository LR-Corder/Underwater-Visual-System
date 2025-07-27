# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import numpy as np
import cv2
import camera_configs

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MARKER_BIG = 500          # mm
MARKER_SMALL = 45         # mm
DICT = cv2.aruco.DICT_7X7_250

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convert Rodrigues vector to Euler angles (degrees)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
dictionary = cv2.aruco.getPredefinedDictionary(DICT)
cap = cv2.VideoCapture(5)

while True:
    ok, frame = cap.read()
    if not ok:
        continue

    corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary)
    if ids is None or 0 not in ids:
        continue

    idx_0 = np.where(ids == 0)[0][0]
    rvec_0, tvec_0, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners[idx_0], MARKER_BIG, camera_configs.left_camera_matrix,
        camera_configs.left_distortion)

    T_0 = np.eye(4)
    T_0[:3, :3], _ = cv2.Rodrigues(rvec_0)
    T_0[:3, 3] = tvec_0.flatten()

    print('----------------- OUTPUT -----------------')
    for i, mid in enumerate(ids.ravel()):
        size = MARKER_BIG if mid == 0 else MARKER_SMALL
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[i], size, camera_configs.left_camera_matrix,
            camera_configs.left_distortion)

        T = np.eye(4)
        T[:3, :3], _ = cv2.Rodrigues(rvec)
        T[:3, 3] = tvec.flatten()

        T_rel = np.linalg.inv(T) @ T_0

        x, y, z = np.round(tvec[0], 1)
        rx, ry, rz = np.round(rvec_to_euler(rvec), 2)

        print(f'aruco_{mid}:')
        print('  relative_pose =', T_rel)
        print('  position =', (x, y, z))
        print('  euler =', (rx, ry, rz))