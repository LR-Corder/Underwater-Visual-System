# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

from __future__ import annotations

import math
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import serial
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import camera_configs_danmu as camera_configs

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SAVE_FLAG = True
SERIAL_PORT = '/dev/ttyTHS1'
SERIAL_BAUD = 115200
VIDEO_SRC = 5
MARKER_BIG = 500
MARKER_SMALL = 55

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class PoseFilter:
    """Exponential smoothing for pose vectors."""

    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.prev_pose: np.ndarray | None = None

    def update(self, pose: np.ndarray) -> np.ndarray:
        if self.prev_pose is None:
            self.prev_pose = pose
            return pose
        self.prev_pose = self.alpha * pose + (1 - self.alpha) * self.prev_pose
        return self.prev_pose


pose_filter = PoseFilter()


def rvec_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convert Rodrigues vector to Euler angles (degrees)."""
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0.0
    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
time.sleep(2)

cap = cv2.VideoCapture(VIDEO_SRC)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

if not cap.isOpened():
    raise RuntimeError('Cannot open camera.')

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Camera: {fps} fps')

if SAVE_FLAG:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = cv2.VideoWriter(
        f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/out-{ts}.avi',
        cv2.VideoWriter_fourcc(*'DIVX'), 9, (640, 480)
    )
    out_3d = cv2.VideoWriter(
        f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_output-{ts}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480)
    )

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)
K = camera_configs.left_camera_matrix
D = camera_configs.left_distortion

# --------------------------------------------------------------------------- #
# 3-D plotting
# --------------------------------------------------------------------------- #
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')
ax.set_title('Camera & Marker')

trajectory: list[np.ndarray] = []


def update_plot(pos: np.ndarray) -> None:
    """Draw marker at (0,0,0) and camera trajectory."""
    ax.cla()
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Camera & Marker')
    ax.scatter(0, 0, 0, c='r', label='Marker')

    trajectory.append(pos)
    traj = np.array(trajectory)
    if traj.shape[0] > 1:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'g-', linewidth=0.5, label='Path')
    ax.scatter(pos[0], pos[1], pos[2], c='b', label='Camera')
    ax.legend()

    fig.canvas.draw()
    if SAVE_FLAG:
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        out_3d.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        plt.pause(0.001)


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
first_flag = False

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break

    img = frame.copy()
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)
    if ids is not None and ids.size:
        first_flag = True
        for mid in ids.ravel():
            size = MARKER_BIG if mid == 29 else MARKER_SMALL
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[np.where(ids == mid)[0][0]], size, K, D
            )
            pos = pose_filter.update(tvec[0, 0])

            x, y, z = np.round(pos, 1)
            rx, ry, rz = map(float, map(lambda a: round(a, 2),
                                        rvec_to_euler(rvec)))

            msg = f'[{x/10:.2f},{y/10:.2f},{abs(z)/10:.2f},{rx:.2f},{ry:.2f},{rz:.2f}]\r\n'
            ser.write(msg.encode())

            cv2.putText(img, msg.strip(), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            update_plot(pos)
            cv2.aruco.drawDetectedMarkers(img, corners)
    else:
        if first_flag:
            ser.write(b'[0,0,0,0,0,0]\r\n')
            cv2.putText(img, 'No IDs', (0, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('out', img)
    if SAVE_FLAG:
        cv2.putText(img, datetime.now().strftime('%Y:%m:%d:%H:%M:%S'),
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        out.write(img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
cap.release()
cv2.destroyAllWindows()
ser.close()
if SAVE_FLAG:
    out.release()
    out_3d.release()
plt.close(fig)