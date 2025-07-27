# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

from datetime import datetime
import time

import cv2
import numpy as np
import serial

import camera_configs

SAVE_FLAG = True


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert 3*3 rotation matrix to quaternion [x, y, z, w]."""
    q_w = np.sqrt(1.0 + np.trace(matrix)) / 2.0
    q_x = (matrix[2, 1] - matrix[1, 2]) / (4 * q_w)
    q_y = (matrix[0, 2] - matrix[2, 0]) / (4 * q_w)
    q_z = (matrix[1, 0] - matrix[0, 1]) / (4 * q_w)
    return np.array([q_x, q_y, q_z, q_w])


class PoseFilter:
    """Simple exponential smoothing for pose vectors."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self.prev_pose = None

    def filter(self, pose: np.ndarray) -> np.ndarray:
        if self.prev_pose is None:
            self.prev_pose = pose
            return pose
        self.prev_pose = self.alpha * pose + (1 - self.alpha) * self.prev_pose
        return self.prev_pose


pose_filter = PoseFilter()


def rvec_to_euler(rvec) -> tuple[float, float, float]:
    """Convert Rodrigues vector to Euler angles (deg)."""
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


# Serial init
ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.1)
time.sleep(2)

# Camera init
cap = cv2.VideoCapture(5)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

if not cap.isOpened():
    raise RuntimeError('Cannot open camera.')

fps = int(cap.get(cv2.CAP_PROP_FPS))
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'Camera: {fps} fps, {w}x{h}')

if SAVE_FLAG:
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out_frame = cv2.VideoWriter(
        f'/home/nvidia/new_rec2/followmanta_session/log_land/out-{time.ctime()}.avi',
        fourcc, 5, (640, 480))

# ArUco
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
K = camera_configs.left_camera_matrix
D = camera_configs.left_distortion

# Marker definitions
ARUCO_0 = np.array([...])  # truncated for brevity
ARUCO_48 = np.array([...])
ARUCO_47 = np.array([...])
ARUCO_11 = np.array([...])
ARUCO_5 = np.array([...])
ARUCO_46 = np.array([...])
ARUCO_45 = np.array([...])
ARUCO_ELSE = np.array([...])

marker_map = {0: ARUCO_0, 48: ARUCO_48, 47: ARUCO_47, 11: ARUCO_11,
              5: ARUCO_5, 46: ARUCO_46, 45: ARUCO_45}

first_flag = False

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break

    img = frame.copy()
    corners, ids, _ = cv2.aruco.detectMarkers(img, dictionary)

    if ids is not None and len(ids):
        first_flag = True
        for i, mid in enumerate(ids.ravel()):
            marker_size = 373 if mid == 0 else 40
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], marker_size, K, D)

            tvec = pose_filter.filter(tvec[0, 0])

            # Transform using predefined marker poses
            T = np.eye(4)
            T[:3, :3], _ = cv2.Rodrigues(rvec)
            T[:3, 3] = tvec
            T = T @ marker_map.get(mid, ARUCO_ELSE)
            tvec = T[:3, 3]

            rx, ry, rz = rvec_to_euler(rvec)
            rx = (rx + 180) % 360 - 180
            ry = (ry + 180) % 360 - 180
            rz = (rz + 180) % 360 - 180

            x, y, z = tvec / 10.0
            msg = f'[{x:.2f},{y:.2f},{abs(z):.2f},{rx:.2f},{ry:.2f},{rz:.2f}]\r\n'
            ser.write(msg.encode())
            cv2.putText(img, msg.strip(), (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        if not first_flag:
            continue
        ser.write(b'[0,0,0,0,0,0]\r\n')
        cv2.putText(img, 'No IDs', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow('out', img)
    if SAVE_FLAG:
        ts = datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
        cv2.putText(img, ts, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        out_frame.write(img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
if SAVE_FLAG:
    out_frame.release()