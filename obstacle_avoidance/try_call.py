import cv2
import obstacle_depth

camera = cv2.VideoCapture(r"C:\Users\h\Downloads\py.project_cv\try\out-frameMon Sep 30 15_33_56 2024.avi")

while True:
    ret , frame = camera.read()
    if not ret:
        break
    frame_with_obstacles = obstacle_depth.frame_obstacles(frame,False,True)
    for obs in frame_with_obstacles.obstacles:
        print(len(obs.contour))
    # print(frame_with_obstacles.obstacles)
    key = cv2.waitKey(30)
    if key == 27:
        break