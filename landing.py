# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

from __future__ import annotations

import math
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import camera_configs

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SAVE_VID = True
VIDEO_SRC = 0
MARKER_BIG = 500
MARKER_SMALL = 55

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class PoseFilter:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.prev: np.ndarray | None = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        if self.prev is None:
            self.prev = pose
            return pose
        self.prev = self.alpha * pose + (1 - self.alpha) * self.prev
        return self.prev


pose_filter = PoseFilter()


def rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """Rodrigues → Euler angles (degrees)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    else:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Camera: {fps} fps')

if SAVE_VID:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = cv2.VideoWriter(
        f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/out-{ts}.avi',
        cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 480)
    )
    out_3d = cv2.VideoWriter(
        '/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480)
    )

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
K = camera_configs.left_camera_matrix
D = camera_configs.left_distortion

records: list[dict[str, float]] = []

# --------------------------------------------------------------------------- #
# 3-D plotting
# --------------------------------------------------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Camera & Marker')
traj: list[np.ndarray] = []


def update_plot(pos: np.ndarray) -> None:
    ax.cla()
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Camera & Marker')
    ax.scatter(0, 0, 0, c='r', label='Marker')

    traj.append(pos)
    t = np.array(traj)
    if t.shape[0] > 1:
        ax.plot(t[:, 0], t[:, 1], t[:, 2], 'g-', linewidth=0.5, label='Path')
    ax.scatter(pos[0], pos[1], pos[2], c='b', label='Camera')
    ax.legend()

    fig.canvas.draw()
    if SAVE_VID:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        out_3d.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
first = False

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break

    img = cv2.flip(frame, -1)
    img = img[:, : img.shape[1] // 2]
    img = cv2.remap(img, camera_configs.left_map1, camera_configs.left_map2,
                    cv2.INTER_LINEAR)

    vis = img.copy()
    corners, ids, _ = cv2.aruco.detectMarkers(vis, dictionary)

    if ids is not None and ids.size:
        first = True
        for mid in ids.ravel():
            size = MARKER_BIG if mid == 29 else MARKER_SMALL
            idx = np.where(ids == mid)[0][0]
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[idx], size, K, D
            )
            pos = pose_filter.update(tvec[0, 0])

            x, y, z = map(float, np.round(pos, 1))
            rx, ry, rz = map(float, map(lambda a: round(a, 2),
                                        rvec_to_euler(rvec)))

            records.append({'Id': mid, 'Position_X': x, 'Position_Y': y,
                            'Position_Z': z, 'Rotation_X': rx,
                            'Rotation_Y': ry, 'Rotation_Z': rz})

            msg = f'[{x/10:.2f},{y/10:.2f},{abs(z)/10:.2f},{rx:.2f},{ry:.2f},{rz:.2f}]\r\n'
            print(msg.strip())
            cv2.putText(vis, msg.strip(), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            update_plot(pos)
            cv2.aruco.drawDetectedMarkers(vis, corners)
    else:
        if first:
            cv2.putText(vis, 'No IDs', (0, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('out', vis)
    if SAVE_VID:
        out.write(vis)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
cap.release()
cv2.destroyAllWindows()
plt.close(fig)

if SAVE_VID:
    out.release()
    out_3d.release()

# Export to Excel
df = pd.DataFrame(records)
df.to_excel('rgb_500_camera_positions.xlsx', index=False)