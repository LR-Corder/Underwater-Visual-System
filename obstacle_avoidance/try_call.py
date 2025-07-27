# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import cv2
import obstacle_depth

# Open video capture
camera = cv2.VideoCapture(r"C:\Users\h\Downloads\py.project_cv\try\out-frameMon Sep 30 15_33_56 2024.avi")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Process frame to detect obstacles
    frame_with_obstacles = obstacle_depth.FrameObstacles(frame, show_depth=False, show_obstacle=True)

    # Print obstacle information
    for obs in frame_with_obstacles.obstacles:
        print(f"Obstacle ID: {obs.id}, Contour Length: {len(obs.contour)}")

    # Display frame
    cv2.imshow("Frame", frame_with_obstacles.disp)

    # Exit on ESC key
    if cv2.waitKey(30) == 27:
        break

# Release resources
camera.release()
cv2.destroyAllWindows()