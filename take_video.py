# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import time

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CAPTURE_ID = -1
FPS = 5.0
RESOLUTION = (640, 640)
OUTPUT_DIR = '/home/nvidia/new_rec2/followmanta_sesssion/log'
FOURCC = cv2.VideoWriter_fourcc(*'DIVX')
WINDOW_NAMES = ('left', 'right')

# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
for name in WINDOW_NAMES:
    cv2.namedWindow(name)

cap = cv2.VideoCapture(CAPTURE_ID)
if not cap.isOpened():
    raise RuntimeError('Cannot open camera.')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0] * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f'Camera info: {width}x{height}, {fps} fps')

timestamp = time.ctime()
left_writer = cv2.VideoWriter(
    f'{OUTPUT_DIR}/left{timestamp}.avi', FOURCC, FPS, (width // 2, height)
)
right_writer = cv2.VideoWriter(
    f'{OUTPUT_DIR}/right{timestamp}.avi', FOURCC, FPS, (width // 2, height)
)

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    left_frame = frame[:, :width // 2]
    right_frame = frame[:, width // 2:]

    left_frame = cv2.flip(left_frame, -1)
    right_frame = cv2.flip(right_frame, -1)

    left_writer.write(left_frame)
    right_writer.write(right_frame)

    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
cap.release()
left_writer.release()
right_writer.release()
cv2.destroyAllWindows()