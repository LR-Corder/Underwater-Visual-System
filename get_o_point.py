# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Contour extraction
# --------------------------------------------------------------------------- #
def extract_contours(image: np.ndarray) -> tuple[np.ndarray, list]:
    """Find outer contours twice; return visualisation and contour list."""
    cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]),
                  (255, 255, 0), 30, 4, 0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 40, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), 5)

    gray_mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    dst = np.zeros_like(mask)
    for cnt in contours:
        color = np.random.randint(0, 255, (3,)).tolist()
        cv2.drawContours(dst, [cnt], -1, color, 5)

    return dst, contours


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class Point3d:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x, self.y, self.z = x, y, z


def _sort_by_area(points: list) -> None:
    """In-place bubble sort on third element (area)."""
    n = len(points)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if points[j][2] > points[j + 1][2]:
                points[j], points[j + 1] = points[j + 1], points[j]
                swapped = True
        if not swapped:
            break


def _compute_centroids(
    dst: np.ndarray,
    contours: list,
    points: list,
    depth: np.ndarray
) -> int:
    """Fill `points` with centroid and bbox info; return count."""
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) <= 1000:
            continue
        temp = np.array(cnt)
        moments = cv2.moments(temp)
        if moments['m00'] == 0:
            continue

        x = int(moments['m10'] / moments['m00'])
        y = int(moments['m01'] / moments['m00'])

        left = tuple(temp[temp[:, :, 0].argmin()][0])
        right = tuple(temp[temp[:, :, 0].argmax()][0])
        top = tuple(temp[temp[:, :, 1].argmin()][0])
        bottom = tuple(temp[temp[:, :, 1].argmax()][0])

        points[count] = (x, y, int(cv2.contourArea(cnt)), left, right, top, bottom)
        cv2.circle(dst, (x, y), 1, (0, 255, 255), 30)
        count += 1
    return count


def _nms_filter(
    canvas: np.ndarray,
    input_points: list,
    output_points: list,
    count: int,
    depth: np.ndarray
) -> tuple[list, list]:
    """Non-max suppression and distance-based classification."""
    targets, objects = [], []
    threshold = 50
    buffer = [Point3d(0, 0, 0) for _ in range(100)]

    idx = 0
    while count:
        buffer[idx] = input_points[count - 1]
        p = buffer[idx]

        left, right, top, bottom = p[3], p[4], p[5], p[6]

        # NMS
        i = 0
        while i < count - 1:
            dist = np.linalg.norm(
                np.array([p[0], p[1]]) - np.array([input_points[i][0], input_points[i][1]])
            )
            if dist < threshold:
                del input_points[i]
                count -= 1
            else:
                i += 1
        count -= 1
        idx += 1

    for i in range(idx):
        x, y = buffer[i][:2]
        area = buffer[i][2]
        length = abs(depth[buffer[i][3][1], buffer[i][3][0], 0] -
                     depth[buffer[i][4][1], buffer[i][4][0], 0])
        width = abs(depth[buffer[i][5][1], buffer[i][5][0], 0] -
                    depth[buffer[i][6][1], buffer[i][6][0], 0])

        if abs(depth[y, x, 2]) < 8:           # target distance filter
            cv2.circle(canvas, (x, y), 1, (0, 0, 255), 30)
            targets.append([x, y, area, length, width])
        else:
            cv2.circle(canvas, (x, y), 1, (0, 255, 255), 30)
            objects.append([x, y, area, length, width])

    cv2.imshow("final_result", canvas)
    return targets, objects


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def detect_targets(src: np.ndarray, depth: np.ndarray) -> list:
    """Main entry: extract targets from RGB and depth."""
    points = [None] * 100
    dst, contours = extract_contours(src)

    count = _compute_centroids(dst, contours, points, depth)
    _sort_by_area(points)

    buffer = [Point3d(0, 0, 0) for _ in range(100)]
    targets, _ = _nms_filter(src, points, buffer, count, depth)
    return targets