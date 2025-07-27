# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import numpy as np
import camera_configs

# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
VIDEO_SRC = 0
PARAMS = {
    "minDisparity": 0,
    "numDisparities": 16 * 4,
    "blockSize": 5,
    "P1": 8 * 3 * 5 * 5,
    "P2": 32 * 3 * 5 * 5,
    "disp12MaxDiff": 0,
    "preFilterCap": 15,
    "uniquenessRatio": 5,
    "speckleWindowSize": 100,
    "speckleRange": 2,
}

# --------------------------------------------------------------------------- #
# Stereo pipeline
# --------------------------------------------------------------------------- #
_left_matcher = cv2.StereoSGBM_create(**PARAMS)
_right_matcher = cv2.ximgproc.createRightMatcher(_left_matcher)

_wls = cv2.ximgproc.createDisparityWLSFilter(_left_matcher)
_wls.setLambda(8000.0)
_wls.setSigmaColor(1.3)
_wls.setLRCthresh(24)
_wls.setDepthDiscontinuityRadius(3)


def _rectify(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remap stereo pair to canonical geometry."""
    l = cv2.remap(left, camera_configs.left_map1, camera_configs.left_map2,
                  cv2.INTER_LINEAR)
    r = cv2.remap(right, camera_configs.right_map1, camera_configs.right_map2,
                  cv2.INTER_LINEAR)
    return l, r


def _compute_depth(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return 3-D point cloud (H×W×3)."""
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    left_disp = _left_matcher.compute(left_gray, right_gray)
    right_disp = _right_matcher.compute(right_gray, left_gray)

    filtered = _wls.filter(left_disp, left_gray,
                           disparity_map_right=right_disp)
    return cv2.reprojectImageTo3D(filtered.astype(np.float32) / 16.0,
                                  camera_configs.Q)


# --------------------------------------------------------------------------- #
# Mouse callback
# --------------------------------------------------------------------------- #
_last_dis = 125  # mm


def _on_click(event: int, x: int, y: int, flags: int, userdata) -> None:
    """Print depth of clicked pixel (with 3×3 neighbour median)."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    neighbours = [(y + dy, x + dx)
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    distances = [threeD[j, i, 2] for j, i in neighbours if 0 <= i < threeD.shape[1] and 0 <= j < threeD.shape[0]]
    if not distances:
        return
    d = np.median(np.abs(distances))
    if d > 10000 or (_last_dis != 125 and abs(d - _last_dis) > 500):
        return
    print(f'{d:.2f}')
    global _last_dis
    _last_dis = d


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
camera = cv2.VideoCapture(VIDEO_SRC)
cv2.namedWindow('depth')
cv2.setMouseCallback('depth', _on_click)

while True:
    ok, frame = camera.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    left = frame[:, w // 2:]
    right = frame[:, :w // 2]

    left_rect, right_rect = _rectify(left, right)
    threeD = _compute_depth(left_rect, right_rect)

    disp = cv2.normalize(cv2.ximgproc.createDisparityWLSFilter(
        _left_matcher).filter(
        _left_matcher.compute(
            cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)),
        cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)),
        None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow('left', left_rect)
    cv2.imshow('right', right_rect)
    cv2.imshow('depth', disp)

    key = cv2.waitKey(50)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite('snapshot_left.jpg', left_rect)
        cv2.imwrite('snapshot_right.jpg', right_rect)
        cv2.imwrite('snapshot_depth.jpg', disp)

camera.release()
cv2.destroyAllWindows()