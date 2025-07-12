# 调用双目摄像头录像，按q退出录制

import cv2
import time

cv2.namedWindow('left')
cv2.namedWindow('right')

cap = cv2.VideoCapture(-1)  # 打开摄像头，摄像头的ID不同设备上可能不同

# cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640*2)  # 设置双目的宽度(整个双目相机的图像宽度)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)  # 设置双目的高度


# 获取视频的帧率和尺寸
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('----------', width, height, fps) #fps:nvidia:5 PC：30


fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # fourcc编码为视频格式，avi对应编码为XVID
left = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log/left{time.ctime()}.avi', fourcc, 5.0, (width//2, height))  # (视频储存位置和名称，储存格式，帧率，视频大小)
right = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log/right{time.ctime()}.avi', fourcc, 5.0, (width//2, height))

while cap.isOpened():
    ret, frame = cap.read()
    print('ret:', ret)

    # left_frame = frame[0:height, 0:width//2]  # 裁剪坐标为[y0:y1，x0：x1]
    # right_frame = frame[0:height, width // 2 : width]

    frame1_ = frame[0 : height, 0 : width // 2] # 左侧相机
    frame2_ = frame[0 : height, width // 2 : 1280] # 右侧相机

    right_frame = cv2.flip(frame2_, -1)  # 旋转180
    left_frame = cv2.flip(frame1_, -1)  # 旋转180

    if not ret:
        break

    left.write(left_frame)
    right.write(right_frame)
    cv2.imshow('left', left_frame)
    cv2.imshow('right', right_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
left.release()
right.release()
cv2.destroyAllWindows()