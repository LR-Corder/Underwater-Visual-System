# v4去掉tcp通信】
import copy
print(1)
import cv2
import numpy as np
import random
import math
import time
from get_o_point import result_point
start = 0
def fill_inf_with_interpolation(threeD):
    # 获取 inf 的位置
    inf_positions = np.where(np.isinf(threeD))

    # 对每个维度进行处理
    for dim in range(threeD.shape[2]):
        valid_positions = np.where(np.isfinite(threeD[:,:,dim]))
        valid_values = threeD[:,:,dim][valid_positions]
        threeD[:,:,dim][inf_positions[0], inf_positions[1]] = np.interp(inf_positions[1], valid_positions[1], valid_values)

    return threeD
def mask_depth(image, threshold=150):
    # 获取图像的行数和列数
    nr, nc = image.shape[:2]
    # 遍历图像的每个像素
    for i in range(nr):
        for j in range(nc):
            # 如果像素值大于阈值，则在th中将该像素设置为0
            if image[i, j] < threshold:
                image[i, j] = 0
    return image

def find_obstacle(depth, thresh=20, max_thresh=255, area=50):

    # 应用遮罩深度函数

    dep =  mask_depth(depth)
    cv2.imshow("depth2", dep)
    # 将深度图像转换为8位单通道图像
    # dep = cv2.convertScaleAbs(dep, alpha=(255.0/65535.0))
    # 显示深度图像
    cv2.imshow("depth3", dep)
    # 创建结构元素
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    # 进行开操作
    out = cv2.morphologyEx(dep, cv2.MORPH_OPEN, element)
    # 显示效果图
    cv2.imshow("opencv4", out)
    # 对图像进行二值化
    _, threshold_output = cv2.threshold(dep, thresh, max_thresh, cv2.THRESH_BINARY)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(threshold_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 对每个轮廓计算其凸包
    hull = [cv2.convexHull(c) for c in contours]
    result = []
    # 绘出轮廓及其凸包
    drawing = np.zeros((threshold_output.shape[0], threshold_output.shape[1], 3), dtype=np.uint8)
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < area:
            continue
        result.append(hull[i])
        color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
        # cv2.drawContours(drawing, contours, i, color, 1, 8)
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    cv2.imshow("contours", drawing)
    return result

def k_means_out(gray):
    # 读取图片
    # image = cv2.imread(r'C:\Users\h\Downloads\py.project_cv\bin_crame\save_new\disparity_map338.png')
    # image  = img

    # 将图片转换为灰度图
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = gray
    # 将灰度图像转换为一维数组
    pixels = gray_image.reshape((-1, 1))

    # 将数据转换为float类型
    pixels = np.float32(pixels)

    # 定义K均值聚类的停止标准
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 设置聚类数量，默认为4
    num_clusters = 3

    # 应用K均值聚类
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将标签重新调整为与图像大小一致的形状
    labels = labels.reshape(gray_image.shape)
    center_out=[]
    # 遍历每个聚类，并打印出块的面积和中心点坐标
    for i in range(num_clusters):
        # 创建一个二值图像，仅包含当前聚类的区域
        cluster_mask = np.uint8(labels == i)
        # 寻找轮廓
        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 计算聚类区域的面积
        area = cv2.countNonZero(cluster_mask)
        # 计算中心点坐标
        moments = cv2.moments(cluster_mask)
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        center_out.append(center)
        # # 在原图上绘制聚类区域轮廓
        # cv2.drawContours(image, contours, -1, (0,255, 0), 2)
        #
        # # 在原图上绘制中心点
        # cv2.circle(image, center, 5, (0, 0, 255), -1)
        # # 在原图上标注块编号
        # cv2.putText(image, f"Target {i+1}", (center[0] + 10, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

        # 打印面积和中心点坐标
        # print(f"块{i + 1}: 面积 = {area}, 中心点坐标 = {center}")

    # # 显示结果
    # cv2.imshow('Result', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return center_out

# -------------------------------------------------------------------
left_camera_matrix = np.array([ [630.6456, 0.2942, 342.7966],
                                [ 0, 630.6821, 292.2809],
                                [ 0, 0, 1]
                                ])
right_camera_matrix = np.array([ [632.1360, 0.4287, 356.9454],
                                 [ 0, 632.1411, 290.7813],
                                 [ 0, 0, 1]
                                 ])
left_distortion = np.array([[ 0.4072,0.0966, -0.0013, -0.0054, -0.1140]])


right_distortion = np.array([[0.3920, 0.1952, -0.0011, 0.0022, -0.4103]])

om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
# R = cv2.Rodrigues(om)[0] # 使用Rodrigues变换将om变换为R
R = np.array([ [ 0.9999, 0.0015, -0.0170],
               [-0.0015 , 1.0000, -0.0001],
               [0.0170, 0.0001, 0.9999]
               ])
T = np.array([170.5653,0.3794, 2.6282]) # 平移关系向量

size = (1280, 480) # 图像尺寸

#为每个摄像头计算立体校正的映射矩阵R1,R2,P1,P2
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
print(Q)

# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
import cv2
import numpy as np
import socket
from struct import  Struct
import struct

HEAD = 0
LENGTH = 1
OBJ_CLASS = 2
ID = 3
X = 4
Y = 5
Z = 6
ROLL = 7
YAW = 8
PITCH =9
CRC = 10
my_struct = Struct("BBBB3I3f")
my_struct2 = Struct("BBBB3I3fB")
my_struct3 = Struct("4B")
STOP = my_struct3.pack(85,2,33,41)
START = my_struct3.pack(85,2,32,46)
id = 0
def transform(data:float):
    MAX = 32767
    if data>0:
        sig = True
    else:
        sig =False
    data = int(abs(data)*100)
    if data<MAX:
        # data = bin(data)[2:]
        if not sig:
            data |= 32768
    else:
        return None
    return data

def crc_8(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = (crc << 1) ^ 0x07
            else:
                crc <<= 1
            crc &= 0xFF
    return int(crc)

def get_data_xyz(id,x,y,z):
    print(f"id:{id} x = {x},y= {y},z ={z}")
    mes = [0]*11
    mes[HEAD] = 85
    mes[LENGTH] = 27
    mes[ID] = id
    mes[OBJ_CLASS] = 1
    mes[X] = transform(x)
    mes[Y] = transform(y)
    mes[Z] = transform(z)
    bin_data = struct.pack(">BBBB6H",*(mes[:-1]))
    mes[CRC] = crc_8(bin_data)
    message = struct.pack(">BBBB6HB",*mes)
    print(len(message))
    return message

def k_means_depth(depth_image):
    # 将深度图像转换为灰度图像
    gray_depth = depth_image

    # 遍历深度图像，获取距离小于5米的像素点的坐标和深度值
    valid_points = []
    for y in range(depth_image.shape[0]):
        for x in range(depth_image.shape[1]):
            # 获取当前像素点的深度值
            depth = depth_image[y][x]
            # 将深度值转换为米
            depth_meters = depth / 1000.0  # 将毫米转换为米
            # 判断深度是否小于5米
            if depth_meters < 5.0:
                valid_points.append((x, y, depth_meters))

    # 如果没有有效点，返回空列表
    if not valid_points:
        return []

    # 将有效点转换为numpy数组格式
    valid_points = np.array(valid_points)

    # 使用K均值聚类算法对有效点进行聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    centers = np.float32(valid_points[:, :2])  # 将有效点作为初始中心点
    _, labels, _ = cv2.kmeans(centers, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将标签重新调整为与图像大小一致的形状
    labels = labels.flatten()

    # 遍历每个聚类，并获取边框和最小深度值
    objects = []
    for i in range(3):
        # 获取属于当前聚类的有效点
        cluster_points = valid_points[labels == i]
        # 计算聚类的外接矩形
        x, y, w, h = cv2.boundingRect(cluster_points[:, :2].astype(np.int32))
        # 获取聚类的深度值
        depth_values = cluster_points[:, 2]
        # 计算最小深度值
        min_depth = np.min(depth_values)
        # 将聚类信息添加到对象列表中
        objects.append({"bbox": (x, y, x + w, y + h), "depth": min_depth})

    return objects


# 加载视频文件
#摄像头实时检测floa
cap = cv2.VideoCapture(r"D:\Users\lihao\Desktop\manta-1\MyVideo-1.mp4")
# cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 480)  #打开并设置摄像头

#导入视频生成深度图
# cap = cv2.VideoCapture("E:/PycharmProjects/PythonStudy/cv-test/stereo_video.avi")
# WIN_NAME = 'Deep disp'
# cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
# --------------------------------------通信-----------------------------------------------------

# -----------------------------------------------------------------------------------------------
# 读取视频
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
while ret:
    # 开始计时
    t1 = time.time()
    # 是否读取到了帧，读取到了则为True
    ret, frame = cap.read()
    # 切割为左右两张图片
    frame1 = frame[0:480, 640:1280]
    frame2 = frame[0:480, 0:640]
    cv2.imshow("frame",frame1)
    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 4
    img_channels = 3
    # stereo = cv2.StereoSGBM_create(minDisparity=1,
    #                                numDisparities=64,
    #                                blockSize=blockSize,
    #                                P1=8 * img_channels * blockSize * blockSize,
    #                                P2=32 * img_channels * blockSize * blockSize,
    #                                disp12MaxDiff=-1,
    #                                preFilterCap=1,
    #                                uniquenessRatio=10,
    #                                speckleWindowSize=100,
    #                                speckleRange=100,
    #                                mode=cv2.STEREO_SGBM_MODE_HH)
    # # 计算视差
    # disparity = stereo.compute(img1_rectified, img2_rectified)
    #
    # # 归一化函数算法，生成深度图（灰度图）
    # disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("disp",disp)


    # 创建深度图窗口
    # cv2.namedWindow("depth", cv2.WINDOW_AUTOSIZE)
    left_matcher = cv2.StereoSGBM_create(minDisparity=1,
                                         numDisparities=64,
                                         blockSize=blockSize,
                                         P1=8 * img_channels * blockSize * blockSize,
                                         P2=32 * img_channels * blockSize * blockSize,
                                         disp12MaxDiff=-1,
                                         preFilterCap=1,
                                         uniquenessRatio=10,
                                         speckleWindowSize=100,
                                         speckleRange=100,
                                         mode=cv2.STEREO_SGBM_MODE_HH)
    #
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    left_disp = left_matcher.compute(imgL, imgR)
    right_disp = right_matcher.compute(imgR, imgL)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    # sigmaColor典型范围值为0.8-2.0
    wls_filter.setLambda(4000.)
    wls_filter.setSigmaColor(2)
    wls_filter.setLRCthresh(30)
    wls_filter.setDepthDiscontinuityRadius(5)

    filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)

    cv2.imshow("fi_disp",filtered_disp)

    # print(f"before:{filtered_disp[100][100]}")
    min_val =0  # 假设最小的感兴趣的深度值为500mm
    max_val =30000  # 假设最大的感兴趣的深度值为3500mm
    # 将深度值限制在[min_val, max_val]范围内
    # clipped_depth = np.clip(filtered_disp, min_val,max_val)
    # print(f"after:{clipped_depth[100][100]}")
    # cv2.imshow("clip_depth", clipped_depth.astype(np.uint16))
    # 正则化深度图像
    # normalized_depth = 255 * (filtered_disp - min_val) / (max_val - min_val)
    # cv2.imshow("norm", normalized_depth)
    # normalized_depth = normalized_depth.astype(np.uint16)
    disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 计算三维坐标数据值
    threeD = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD1 = threeD * 16/1000
    threeD1 = fill_inf_with_interpolation(threeD1)
    disp2 = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    target = result_point(disp2, threeD1)
    for point in target:
        print("point: ",point)
        id += 1
        if id > 240: id = 0
        x, y, z = threeD1[point[1]][point[0]]
        o_class = 1
        # if point[2]<10000 else 2
        # print(f"x:{x},y:{y},z:{z}")
        # chang = np.linalg.norm(point[3])
        # kuan = np.linalg.norm(point[3])
        chang = point[3][0]
        kuan = point[3][1]
        print(f"x:{x},y:{y},z:{z},chang:{chang},kuan:{kuan}")
        #data = get_data_xyz( id, x, y, z)
    # cv2.imshow("norm",normalized_depth)
    cv2.imshow("opencv_norm",disp)

    # find_obstacle(normalized_depth)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

# 释放资源
cap.release()

# 关闭所有窗口
cv2.destroyAllWindows()
