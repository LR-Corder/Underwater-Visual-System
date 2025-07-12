# import cv2
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import camera_configs_danmu as camera_configs
# import time
# from matplotlib import animation
# from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

# import serial
# # save_flag = False
# save_flag = True
# # 定义一个简单的滤波器类来平滑位姿估计结果
# class PoseFilter:
#     def __init__(self, alpha=0.2):
#         self.alpha = alpha
#         self.prev_pose = None

#     def filter_pose(self, new_pose):
#         if self.prev_pose is None:
#             self.prev_pose = new_pose
#             return new_pose
#         else:
#             filtered_pose = self.alpha * new_pose + (1 - self.alpha) * self.prev_pose
#             self.prev_pose = filtered_pose
#             return filtered_pose

# # 初始化滤波器对象
# pose_filter = PoseFilter()


# def rotationVectorToEulerAngles(rvec):
#     R = np.zeros((3, 3), dtype=np.float64)
#     cv2.Rodrigues(rvec, R)
#     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     singular = sy < 1e-6
#     if not singular:
#         x = math.atan2(R[2, 1], R[2, 2])
#         y = math.atan2(-R[2, 0], sy)
#         z = math.atan2(R[1, 0], R[0, 0])
#     else:
#         x = math.atan2(-R[1, 2], R[1, 1])
#         y = math.atan2(-R[2, 0], sy)
#         z = 0
#     return x * 180.0 / np.pi, y * 180.0 / np.pi, z * 180.0 / np.pi


# # 初始化串口通信
# ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.1)  # 修改串口端口和波特率
# time.sleep(2) # 等待串口初始化


# # 打开摄像头
# input_video = cv2.VideoCapture(5)
# # 设置缓存区的大小 !!!
# input_video.set(cv2.CAP_PROP_BUFFERSIZE,1)
# input_video.set(cv2.CAP_PROP_FPS, 20)
# input_video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
# input_video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
# if not input_video.isOpened():
#     print("Error: Could not open video.")
#     exit()

# # 获取视频的帧率和尺寸
# fps = int(input_video.get(cv2.CAP_PROP_FPS))
# frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# print('camera info:', fps, frame_width, frame_height)

# if save_flag:
#     # output = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log/out-remapframe{time.ctime()}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 480))
#     # plot_frame = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/out-plot{time.ctime()}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 480), False)
#     output_frame = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/out-frame{time.ctime()}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 480))
#     output_video_3d = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_output{time.ctime()}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))  # 3D绘图视频





# # 获取ArUco字典
# dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

# # # 左相机内参数矩阵和畸变系数
# # camera_matrix_left = np.array([[400.931244, 0, 347.495483],
# #                                [0, 401.846680, 246.095093],
# #                                [0, 0, 1]])
# # dist_coeff_left = np.array([[0.00165131455, -0.0108984737, 0.00100589055, 0.00167077256, 0.000904905959]])
# # -------------
# # 左相机的内参矩阵  
# camera_matrix_left  = camera_configs.left_camera_matrix
  
# # 左相机畸变系数  
# dist_coeff_left = camera_configs.left_distortion

# # --------------
# # 创建一个空的数据框，用于存储相机位置信息
# columns = ['Id', 'Position_X', 'Position_Y', 'Position_Z', 'Rotation_X', 'Rotation_Y', 'Rotation_Z']
# camera_positions_df = pd.DataFrame(columns=columns)

# # 创建一个3D绘图窗口
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('X Position (mm)')
# ax.set_ylabel('Y Position (mm)')
# ax.set_zlabel('Z Position (mm)')
# ax.set_title('Camera and Marker 3D Position')

# # 存储相机位置的列表，用于绘制轨迹
# camera_trajectory = []
# width = 1280
# height = 480
# def init():
#     ax.cla()  # 清空当前的图形
#     ax.set_xlabel('X Position (mm)')
#     ax.set_ylabel('Y Position (mm)')
#     ax.set_zlabel('Z Position (mm)')
#     ax.set_title('Camera and Marker 3D Position')
# # camera_position, marker_position
# def update_plot(camera_position, marker_position):
#     ax.cla()  # 清空当前的图形
#     ax.set_xlabel('X Position (mm)')
#     ax.set_ylabel('Y Position (mm)')
#     ax.set_zlabel('Z Position (mm)')
#     ax.set_title('Camera and Marker 3D Position')

#     # 绘制固定的标记位置 (假设在[0, 0, 0])
#     ax.scatter(marker_position[0], marker_position[1], marker_position[2], c='r', label='Marker')

#     # 记录相机位置并绘制轨迹
#     camera_trajectory.append(camera_position)
#     trajectory = np.array(camera_trajectory)

#     if trajectory.shape[0] > 1:
#         line, = ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='g', linewidth=0.5,
#                 label='Camera Trajectory')

#     # 绘制相机位置
#     ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='b', label='Camera')

#     ax.legend()
#     plt.draw()
#     if save_flag:
#         # 将当前绘图保存为图像帧
#         fig.canvas.draw()
#         img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         output_video_3d.write(img)
#     else:
#         # plt.draw()
#         plt.pause(0.001)  # 暂停以更新绘图
#     # # 将当前帧写入视频文件
#     # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     # plot_frame.write(frame_rgb)
#     # anim.save('/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_animation.mp4', writer='ffmpeg', fps=30)
# first_flag = False

# # 主循环
# while True:
#     # 读取摄像头帧
#     ret, frame = input_video.read()
#     if frame is None or not ret:
#         print("错误：无法读取帧。")
#         break

#     # frame = cv2.flip(frame, -1) 
#     # frame = frame[0 : height, 0 : width // 2]


#     # frame2 = frame[0 : height, 0 : width // 2] # 右侧相机
#     # frame = frame[0 : height, width // 2 : 1280] # 左侧相机

    


#     # frame1_ = frame[0 : height, 0 : width // 2] # 左侧相机
#     # frame2_ = frame[0 : height, width // 2 : 1280] # 右侧相机

#     # frame2 = cv2.flip(frame2_, -1)  # 旋转180
#     # frame1 = cv2.flip(frame1_, -1)  # 旋转180





#     # frame = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
#     # 复制一份图像用于绘制
#     image_copy = frame.copy()
#     corners, ids, _ = cv2.aruco.detectMarkers(image_copy, dictionary)
#     cnt = 0
#     if ids is not None and len(ids) > 0:
#         print(cnt)
#         cnt += 1
#         first_flag = True
#         for i in range(len(ids)):
#             aruco_info = {'Id': int(ids[i].item())}
#             print(aruco_info['Id'])
#             # 设置标记大小
#             if aruco_info['Id'] == 29:
#                 marker_size = 500
#             elif aruco_info['Id'] == 33:
#                 marker_size = 55
#             else:
#                 marker_size = 0

#             if marker_size > 0:
#                 rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix_left,
#                                                                     dist_coeff_left)
#                 filtered_tvec = pose_filter.filter_pose(tvec[0][0])

#                 # 更新数据框
#                 aruco_info[f'Position_X'] = round(float(tvec[0][0][0]), 1)
#                 aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]), 1)
#                 aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
#                 aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
#                 aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
#                 aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
#                 # print(f"distance y----- {aruco_info[f'Position_Y']  }")
#                 # print(f"distance x ----- {aruco_info[f'Position_X'] }")
#                 # print(f"distance z----- {aruco_info[f'Position_Z']  }")
#                 print(f"发送位置数据：[ positon_x :{aruco_info[f'Position_X']} , positon_y :{aruco_info[f'Position_Y']} , positon_z :{aruco_info[f'Position_Z']} ]")
#                 print(f"发送jaodu数据：[ Rotation_X :{aruco_info[f'Rotation_X']} , Rotation_Y :{aruco_info[f'Rotation_Y']} , Rotation_Z :{aruco_info[f'Rotation_Z']} ]")
#                 x_serial = "%.2f"%(aruco_info[f'Position_X']/10)
#                 y_serial="%.2f"%(aruco_info[f'Position_Y']/10)
#                 z_serial="%.2f"%(abs(aruco_info[f'Position_Z']/10))
#                 x_angle_serial = "%.2f"%(aruco_info[f'Rotation_X'])
#                 y_angle_serial = "%.2f"%(aruco_info[f'Rotation_Y'])
#                 z_angle_serial = "%.2f"%(aruco_info[f'Rotation_Z'])
#                 serial_data= ('[' + x_serial + ','+ y_serial+','+z_serial+',' + x_angle_serial + ',' + y_angle_serial + ',' + z_angle_serial +']\r\n')
#                 print("------------")
#                 print(serial_data)
#                 ser.write(serial_data.encode('utf-8'))
#                 cv2.putText(image_copy, f"{serial_data}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
#                 # cv2.putText(output_frame, f"y: {int(centerpoint[1])}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            

#                 # 将信息添加到数据框
#                 camera_positions_df = pd.concat([camera_positions_df, pd.DataFrame([aruco_info])], ignore_index=True)

#                 # 更新绘图，固定标记位置为 [0, 0, 0]
#                 update_plot(filtered_tvec, [0, 0, 0])  # 相机位置变化，标记位置不变
#                 # anim = animation.FuncAnimation(fig, update_plot, frames=3600, interval=20, blit=False)
#                 # anim.save('/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_animation.mp4',  fps=30, writer='ffmpeg')
                
#                 # 绘制标记和坐标轴
#                 cv2.aruco.drawDetectedMarkers(image_copy, corners)

#     else:
#         if not first_flag: continue
#         serial_data= ('[' + '0' + ','+ '0'+','+'0'+',' + '0' + ',' + '0' + ',' + '0' +']\r\n')
#         print(serial_data)
#         ser.write(serial_data.encode('utf-8'))
#         cv2.putText(image_copy, "No Ids", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    

#     # 显示图像
#     cv2.imshow("out", image_copy)
#     if save_flag:
#         output_frame.write(image_copy)
   

#     # 等待用户按键，按ESC键退出
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# # 保存数据框到Excel
# excel_file_path = 'D:/BaiduSyncdisk/仿蝠鲼项目/rgb_500_camera_positions.xlsx'
# camera_positions_df.to_excel(excel_file_path, index=False)

# # 释放摄像头资源
# input_video.release()
# cv2.destroyAllWindows()
# output_video_3d.release()
# plt.close(fig)  # 关闭3D绘图窗口/home/nvidia/new_rec2/followmanta_sesssion/log_land/




from datetime import datetime

import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import camera_configs_danmu as camera_configs
import time
from matplotlib import animation
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import serial

# save_flag = False
save_flag = True


# 定义一个简单的滤波器类来平滑位姿估计结果
class PoseFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.prev_pose = None

    def filter_pose(self, new_pose):
        if self.prev_pose is None:
            self.prev_pose = new_pose
            return new_pose
        else:
            filtered_pose = self.alpha * new_pose + (1 - self.alpha) * self.prev_pose
            self.prev_pose = filtered_pose
            return filtered_pose


# 初始化滤波器对象
pose_filter = PoseFilter()


def rotationVectorToEulerAngles(rvec):
    R = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(rvec, R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x * 180.0 / np.pi, y * 180.0 / np.pi, z * 180.0 / np.pi


# 初始化串口通信
ser = serial.Serial('/dev/ttyTHS1', 115200, timeout=0.1)  # 修改串口端口和波特率
time.sleep(2)  # 等待串口初始化

# 打开摄像头
input_video = cv2.VideoCapture(5)
# 设置缓存区的大小 !!!
input_video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
input_video.set(cv2.CAP_PROP_FPS, 20)
input_video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
input_video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
if not input_video.isOpened():
    print("Error: Could not open video.")
    exit()

# 获取视频的帧率和尺寸
fps = int(input_video.get(cv2.CAP_PROP_FPS))
frame_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('camera info:', fps, frame_width, frame_height)

if save_flag:
    # output = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log/out-remapframe{time.ctime()}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 480))
    # plot_frame = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/out-plot{time.ctime()}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 480), False)
    output_frame = cv2.VideoWriter(f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/out-frame{time.ctime()}.avi',
                                   cv2.VideoWriter_fourcc(*'DIVX'), 9, (640, 480))
    output_video_3d = cv2.VideoWriter(
        f'/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_output{time.ctime()}.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'), fps, (640, 480))  # 3D绘图视频

# 获取ArUco字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

# # 左相机内参数矩阵和畸变系数
# camera_matrix_left = np.array([[400.931244, 0, 347.495483],
#                                [0, 401.846680, 246.095093],
#                                [0, 0, 1]])
# dist_coeff_left = np.array([[0.00165131455, -0.0108984737, 0.00100589055, 0.00167077256, 0.000904905959]])
# -------------
# 左相机的内参矩阵
camera_matrix_left = camera_configs.left_camera_matrix

# 左相机畸变系数
dist_coeff_left = camera_configs.left_distortion

# --------------
# 创建一个空的数据框，用于存储相机位置信息
columns = ['Id', 'Position_X', 'Position_Y', 'Position_Z', 'Rotation_X', 'Rotation_Y', 'Rotation_Z']
camera_positions_df = pd.DataFrame(columns=columns)

# 创建一个3D绘图窗口
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Position (mm)')
ax.set_ylabel('Y Position (mm)')
ax.set_zlabel('Z Position (mm)')
ax.set_title('Camera and Marker 3D Position')

# 存储相机位置的列表，用于绘制轨迹
camera_trajectory = []
width = 1280
height = 480


def init():
    ax.cla()  # 清空当前的图形
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_zlabel('Z Position (mm)')
    ax.set_title('Camera and Marker 3D Position')


# camera_position, marker_position
def update_plot(camera_position, marker_position):
    ax.cla()  # 清空当前的图形
    ax.set_xlabel('X Position (mm)')
    ax.set_ylabel('Y Position (mm)')
    ax.set_zlabel('Z Position (mm)')
    ax.set_title('Camera and Marker 3D Position')

    # 绘制固定的标记位置 (假设在[0, 0, 0])
    ax.scatter(marker_position[0], marker_position[1], marker_position[2], c='r', label='Marker')

    # 记录相机位置并绘制轨迹
    camera_trajectory.append(camera_position)
    trajectory = np.array(camera_trajectory)

    if trajectory.shape[0] > 1:
        line, = ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color='g', linewidth=0.5,
                        label='Camera Trajectory')

    # 绘制相机位置
    ax.scatter(camera_position[0], camera_position[1], camera_position[2], c='b', label='Camera')

    ax.legend()
    plt.draw()
    if save_flag:
        # 将当前绘图保存为图像帧
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        output_video_3d.write(img)
    else:
        # plt.draw()
        plt.pause(0.001)  # 暂停以更新绘图
    # # 将当前帧写入视频文件
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # plot_frame.write(frame_rgb)
    # anim.save('/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_animation.mp4', writer='ffmpeg', fps=30)


first_flag = False

# 主循环
while True:
    # 读取摄像头帧
    ret, frame = input_video.read()
    if frame is None or not ret:
        print("错误：无法读取帧。")
        break

    # frame = cv2.flip(frame, -1)
    # frame = frame[0 : height, 0 : width // 2]

    # frame2 = frame[0 : height, 0 : width // 2] # 右侧相机
    # frame = frame[0 : height, width // 2 : 1280] # 左侧相机

    # frame1_ = frame[0 : height, 0 : width // 2] # 左侧相机
    # frame2_ = frame[0 : height, width // 2 : 1280] # 右侧相机

    # frame2 = cv2.flip(frame2_, -1)  # 旋转180
    # frame1 = cv2.flip(frame1_, -1)  # 旋转180

    # frame = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    # 复制一份图像用于绘制
    image_copy = frame.copy()
    corners, ids, _ = cv2.aruco.detectMarkers(image_copy, dictionary)
    cnt = 0
    if ids is not None and len(ids) > 0:
        print(cnt)
        cnt += 1
        first_flag = True
        for i in range(len(ids)):
            aruco_info = {'Id': int(ids[i].item())}
            print(aruco_info['Id'])
            # 设置标记大小
            if aruco_info['Id'] == 29:
                marker_size = 500
            elif aruco_info['Id'] == 33:
                marker_size = 55
            else:
                marker_size = 0

            if marker_size > 0:
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, camera_matrix_left,
                                                                    dist_coeff_left)
                filtered_tvec = pose_filter.filter_pose(tvec[0][0])

                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]), 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]), 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                # print(f"distance y----- {aruco_info[f'Position_Y']  }")
                # print(f"distance x ----- {aruco_info[f'Position_X'] }")
                # print(f"distance z----- {aruco_info[f'Position_Z']  }")
                print(
                    f"发送位置数据：[ positon_x :{aruco_info[f'Position_X']} , positon_y :{aruco_info[f'Position_Y']} , positon_z :{aruco_info[f'Position_Z']} ]")
                print(
                    f"发送jaodu数据：[ Rotation_X :{aruco_info[f'Rotation_X']} , Rotation_Y :{aruco_info[f'Rotation_Y']} , Rotation_Z :{aruco_info[f'Rotation_Z']} ]")
                x_serial = "%.2f" % (aruco_info[f'Position_X'] / 10)
                y_serial = "%.2f" % (aruco_info[f'Position_Y'] / 10)
                z_serial = "%.2f" % (abs(aruco_info[f'Position_Z'] / 10))
                x_angle_serial = "%.2f" % (aruco_info[f'Rotation_X'])
                y_angle_serial = "%.2f" % (aruco_info[f'Rotation_Y'])
                z_angle_serial = "%.2f" % (aruco_info[f'Rotation_Z'])
                serial_data = (
                            '[' + x_serial + ',' + y_serial + ',' + z_serial + ',' + x_angle_serial + ',' + y_angle_serial + ',' + z_angle_serial + ']\r\n')
                print("------------")
                print(serial_data)
                ser.write(serial_data.encode('utf-8'))
                cv2.putText(image_copy, f"{serial_data}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # cv2.putText(output_frame, f"y: {int(centerpoint[1])}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 将信息添加到数据框
                camera_positions_df = pd.concat([camera_positions_df, pd.DataFrame([aruco_info])], ignore_index=True)

                # 更新绘图，固定标记位置为 [0, 0, 0]
                update_plot(filtered_tvec, [0, 0, 0])  # 相机位置变化，标记位置不变
                # anim = animation.FuncAnimation(fig, update_plot, frames=3600, interval=20, blit=False)
                # anim.save('/home/nvidia/new_rec2/followmanta_sesssion/log_land/3d_animation.mp4',  fps=30, writer='ffmpeg')

                # 绘制标记和坐标轴
                cv2.aruco.drawDetectedMarkers(image_copy, corners)

    else:
        if not first_flag: continue
        serial_data = ('[' + '0' + ',' + '0' + ',' + '0' + ',' + '0' + ',' + '0' + ',' + '0' + ']\r\n')
        print(serial_data)
        ser.write(serial_data.encode('utf-8'))
        cv2.putText(image_copy, "No Ids", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示图像
    cv2.imshow("out", image_copy)
    if save_flag:
        # 获取当前时间
        now = datetime.now()

        # 格式化时间为 year:month:day:hour:min:s 的格式
        formatted_time = f"{now.year}:{now.month}:{now.day}:{now.hour}:{now.minute}:{now.second}"
        cv2.putText(image_copy, formatted_time, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        output_frame.write(image_copy)

    # 等待用户按键，按ESC键退出
    key = cv2.waitKey(1)
    if key == 27:
        break

# 保存数据框到Excel
excel_file_path = 'D:/BaiduSyncdisk/仿蝠鲼项目/rgb_500_camera_positions.xlsx'
camera_positions_df.to_excel(excel_file_path, index=False)

# 释放摄像头资源
input_video.release()
cv2.destroyAllWindows()
output_video_3d.release()
plt.close(fig)  # 关闭3D绘图窗口/home/nvidia/new_rec2/followmanta_sesssion/log_land/

