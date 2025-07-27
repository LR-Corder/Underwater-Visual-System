# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import math
import cv2
import numpy as np
import serial
from obstacle_depth import FrameObstacles  # Assuming this is a custom module

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CAMERA_SRC = r"D:/work/北交大/bizhang/WIN_20241224_10_05_16_Pro.mp4"
FOV = 60  # Field of view in degrees
SERIAL_PORT = "/dev/ttyTHS1"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 0.1  # seconds

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
class Point:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def angle_between_points(a: Point, b: Point) -> float:
    """Calculate angle between two points in degrees."""
    dx = b.x - a.x
    dy = b.y - a.y
    angle = math.atan2(dy, dx) * 180 / math.pi - 90
    if angle < 0:
        angle += 360
    return angle


def is_point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """PNPOLY algorithm to check if a point is inside a polygon."""
    x, y = point
    n = len(polygon)
    inside = False
    x_intersections = 0

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        if min(y1, y2) < y <= max(y1, y2):
            if x1 == x2:  # Horizontal edge
                x_intersect = x1
            else:
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

            if x_intersect >= x:
                x_intersections += 1

    return x_intersections % 2 == 1


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Initialize serial communication
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)

    # Open video source
    cap = cv2.VideoCapture(CAMERA_SRC)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source.")

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame to detect obstacles
        frame_with_obstacles = FrameObstacles(frame, show_depth=True, show_obstacle=True)

        # Display processed frame
        cv2.imshow("frame", frame_with_obstacles.disp)

        # Check for key press
        if cv2.waitKey(1) & 0xFF == ord("w"):
            cv2.waitKey(0)

        # Process obstacles
        num_obstacles = frame_with_obstacles.frame_obstacles_size
        if num_obstacles:
            for i in range(num_obstacles):
                obstacle = frame_with_obstacles.obstacles[i]
                center = obstacle.center_xyz_world

                # Check if obstacle is within range
                if abs(center[2]) < 0.3:  # Depth threshold
                    if center[0] < 0:  # Right side
                        if is_point_in_polygon((-0.6, 0), obstacle.contour):
                            # Calculate avoidance angle
                            angle = angle_between_points(Point(0, 0, 0), Point(center[0], center[1], 0))
                            print(f"Avoidance angle: {angle:.2f} degrees")
                            # Send angle via serial
                            ser.write(f"{angle}\r\n".encode())
                    else:  # Left side
                        if is_point_in_polygon((0.6, 0), obstacle.contour):
                            angle = angle_between_points(Point(0, 0, 0), Point(center[0], center[1], 0))
                            print(f"Avoidance angle: {angle:.2f} degrees")
                            ser.write(f"{angle}\r\n".encode())

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    ser.close()