# -*- coding: UTF-8 -*-
import math
import cv2
import obstacle_depth
import serial
import numpy as np


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# 计算点A到点B的角度
def angle_between_points(a, b):
    dx = b.x - a.x
    dy = b.y - a.y
    angle = math.atan2(dy, dx) * 180 / math.pi - 90
    if angle < 0:
        angle += 360
    return angle


# 计算目标返回的矩阵中每一个点的距离值大小
def calculate_pixel_distance(point_deep, fov):
    angle_per_pixel = fov / 64
    length = point_deep * math.tan(math.radians(angle_per_pixel))
    return length


def is_point_in_polygon(point, polygon):
    """
    判断一个点是否在多边形内部（PNPOLY算法）

    :param point: 待判断的点，格式为 (x, y)
    :param polygon: 多边形的顶点，格式为 [(x1, y1), (x2, y2), ..., (xn, yn)]
    :return: 如果点在多边形内部，则返回True，否则返回False
    """
    x, y = point
    n = len(polygon)
    inside = False

    x_intersections = 0
    for i in range(n):
        x1, y1,z1 = polygon[i]
        x2, y2,z1 = polygon[(i + 1) % n]

        if min(y1, y2) < y <= max(y1, y2):
            if x1 == x2:  # 水平边
                x_intersect = x1
            else:
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)

            if x_intersect >= x:
                x_intersections += 1

    inside = (x_intersections % 2 == 1)
    return inside


obstacle_point = {}
obstacle_area = {}
obstacle_center = {}
check_multi_number = 0
pos_degree_list = []

# global frame
# camera = cv2.VideoCapture(-1)
# camera = cv2.VideoCapture(r"./wall.mp4")
# camera = cv2.VideoCapture(r"./test_old.webm")
camera = cv2.VideoCapture(r"D:/work/北交大/bizhang/WIN_20241224_10_05_16_Pro.mp4")
# camera = cv2.VideoCapture(r"./WIN_20241224_10_08_56_Pro.mp4")
# camera = cv2.VideoCapture(r"./WIN_20241224_10_10_27_Pro.mp4")


# "/home/nvidia/new_rec2/followmanta_sesssion/log/out-frameSun Sep 29 21:52:44 2024.avi"


# 设置缓存区的大小 !!!
camera.set(cv2.CAP_PROP_BUFFERSIZE,1)
camera.set(cv2.CAP_PROP_FPS, 20)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('m', 'j', 'p', 'g'))
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))


# 获取视频的帧率和尺寸
fps = camera.get(cv2.CAP_PROP_FPS)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

print('----------', width, height, fps) #fps:nvidia:5 PC：30




def main():
    while True:
        ret, frame = camera.read()
        
        #cv2.imshow("frame",frame)

        
        
        if not ret:
            break
        frame_with_obstacles = obstacle_depth.frame_obstacles(frame, True, True)
        
        if cv2.waitKey(1) & 0xFF == ord("w"):
            cv2.waitKey(0)

        num_obstacle = frame_with_obstacles.frame_obstacles_size  # 障碍物数量
        print('obstacles info',frame_with_obstacles, num_obstacle)
        # continue
        
        if num_obstacle:
            for i in range(num_obstacle):  # 获取每一帧中所需的障碍物信息
                # print(frame_with_obstacles.obstacles[i])
                obstacle_point[i] = frame_with_obstacles.obstacles[i]
                obstacle_area[i] = obstacle_point[i].contour
                obstacle_center[i] = obstacle_point[i].center_xyz_world

            for i in range(num_obstacle):  # 在此进行避障操作,目前只考虑定深下的避障，不结合深度变化
                
                if obstacle_point[i].center_xyz_world[1] < 0:  # 小于0即障碍物中心点在当前位置的右边，此时需要向右躲避
                    if is_point_in_polygon([-0.6, 0], obstacle_area[i]):  # 鱼自身鱼撞上障碍物区域，进行避障操作
                        min_avoid_x = -0.6  # 鱼自身所在的最小位置
                        for j in range(len(obstacle_area[i])):
                            
                            min_avoid_x = max(obstacle_area[i][j][0], min_avoid_x)
                        pos_degree_list.append(angle_between_points(Point(0, 0, 0), Point(min_avoid_x, obstacle_center[0][1], 0)))
                else:
                    if is_point_in_polygon([0.6, 0], obstacle_area[i]):  # 鱼自身鱼撞上障碍物区域，进行避障操作
                        min_avoid_x = 0.6  # 鱼自身所在的最小位置
                        for j in range(len(obstacle_area[i])):
                            min_avoid_x = min(obstacle_area[i][j][0], min_avoid_x)
                        pos_degree_list.append(angle_between_points(Point(0, 0, 0), Point(min_avoid_x, obstacle_center[0][1], 0)))
            
            print('000', pos_degree_list)

            if(len(pos_degree_list)!=0):
                # 进行避障转弯角度计算
                ser_degree = 0
                # 使用列表推导式检查是否有0-60之间的数字
                has_numbers_in_range1 = any(0 <= num <= 60 for num in pos_degree_list)
                # 使用列表推导式检查是否有300-360之间的数字
                has_numbers_in_range2 = any(300 <= num <= 360 for num in pos_degree_list)
                if has_numbers_in_range1 and has_numbers_in_range2:
                    ser_degree = 180
                    # 此处增加串口发送主控数据
                elif 300 < min(pos_degree_list) < 360:
                    ser_degree = min(pos_degree_list)
                elif 0 < max(pos_degree_list) < 60:
                    ser_degree = max(pos_degree_list)
                else:
                    ser_degree = 0
                # 此处增加串口发送主控数据
                print('航向角:',ser_degree)
                pos_degree_list.clear()



'''
    if z_distance - width / 2 < 0.3 or z_distance + width / 2 > -0.3:  # 定深情况下，判断障碍物是否在鱼游动深度范围内，不在则忽略
        if x_distance - length / 2 > 1.2 or x_distance + length / 2 < -1.2:  # 判断该障碍物是否在游动的航向上，若不在则可忽略
            if x_distance > 0:  # 若中心点在坐标系右侧，所用坐标系为右手定则，z轴朝上，x轴朝右
                y_pos = y_distance
                x_pos = x_distance - length / 2 - 1.2
                pos_degree = angle_between_points(Point(0, 0, 0), Point(x_pos, y_pos, 0))
            else:
                y_pos = y_distance
                x_pos = x_distance + length / 2 + 1.2
                pos_degree = angle_between_points(Point(0, 0, 0), Point(x_pos, y_pos, 0))
            zdistance = -1 * 0  # 输出形式，向上为正，向下为负
            # 将角度转换为字符串
            degree_str = str(pos_degree)
            zdistance_str = str(zdistance)
            ser_alldata = '[' + degree_str + ',' + zdistance_str + ']\r\n'
'''


if __name__ == "__main__":
    main()