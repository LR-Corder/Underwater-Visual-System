# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import serial
from ultralytics import YOLO

import camera_configs

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SAVE_FLAG = True
SERIAL_PORT = '/dev/ttyTHS1'
SERIAL_BAUD = 115200
MODEL_PATH = '/home/nvidia/new_rec2/followmanta_sesssion/best-1024ls.pt'
LOG_DIR = Path('/home/nvidia/new_rec2/followmanta_sesssion/log')

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def get_depth_local(left: np.ndarray, right: np.ndarray, x: float, y: float) -> np.ndarray:
    """Compute 3-D point cloud from stereo pair."""
    left_r = cv2.remap(left, *camera_configs.left_maps, cv2.INTER_LINEAR)
    right_r = cv2.remap(right, *camera_configs.right_maps, cv2.INTER_LINEAR)

    img_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
    img_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=8 * 3 * 5 * 5,
        P2=32 * 3 * 5 * 5,
        disp12MaxDiff=0,
        preFilterCap=15,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=2,
    )

    left_disp = stereo.compute(img_l, img_r)
    right_disp = cv2.ximgproc.createRightMatcher(stereo).compute(img_r, img_l)

    wls = cv2.ximgproc.createDisparityWLSFilter(stereo)
    wls.setLambda(8000.0)
    wls.setSigmaColor(1.3)
    filtered = wls.filter(left_disp, img_l, disparity_map_right=right_disp)

    return cv2.reprojectImageTo3D(filtered.astype(np.float32) / 16.0, camera_configs.Q)

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.1)
    time.sleep(2)

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(-1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        raise RuntimeError('Cannot open camera.')

    if SAVE_FLAG:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out = cv2.VideoWriter(
            str(LOG_DIR / f'out-{ts}.avi'),
            cv2.VideoWriter_fourcc(*'DIVX'),
            5,
            (640, 480),
        )

    recent = deque(maxlen=10)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([128, 128, 128], dtype=np.uint8)

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.resize(frame, (1280, 480), interpolation=cv2.INTER_AREA)
        left_raw = frame[:, :640]
        right_raw = frame[:, 640:]
        left = cv2.flip(left_raw, -1)
        right = cv2.flip(right_raw, -1)

        results = model.track(left, persist=True, stream=True, conf=0.7)
        results = list(results)
        if results[0].boxes.id is None:
            ser.write(b'[0,0,0,0,0,0]\r\n')
            annotated = left
        else:
            boxes = results[0].boxes.xywh.cpu().numpy()
            for box in boxes:
                x, y, w, h = box
                roi = left[int(y - h / 2) : int(y + h / 2), int(x - w / 2) : int(x + w / 2)]
                mask = cv2.inRange(roi, lower_black, upper_black)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    M = cv2.moments(cnt)
                    cx = int(M["m10"] / M["m00"]) + int(x - w / 2)
                    cy = int(M["m01"] / M["m00"]) + int(y - h / 2)

                    pts = get_depth_local(left, right, cx, cy)
                    cam_x, cam_y, cam_z = pts[cy, cx] / 10.0  # mm -> cm
                    cam_z = abs(cam_z)

                    recent.append(cam_z)
                    if len(recent) >= 3:
                        cam_z = np.median(recent)

                    dist = np.linalg.norm([cam_x, cam_y, cam_z])
                    msg = f'[{cam_x:.2f},{cam_y:.2f},{cam_z:.2f},{dist:.2f},0,0]\r\n'
                    ser.write(msg.encode())

            annotated = results[0].plot()

        cv2.imshow('YOLOv8 Tracking', annotated)
        if SAVE_FLAG:
            out.write(annotated)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    ser.close()
    if SAVE_FLAG:
        out.release()


if __name__ == '__main__':
    main()