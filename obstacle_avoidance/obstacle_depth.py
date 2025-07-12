import math
from cmath import sqrt
from dataclasses import dataclass
import cv2
import numpy as np
import camera_configs
distance_threshold  = 1000 # mm
@dataclass
class obstacle:
    id: int
    area: float
    center_xy_img: tuple
    center_xyz_world: tuple
    contour_img: list
    contour: list
    angele: float


class frame_obstacles:
    def __init__(self, frame,show_depth = False, show_obstacle = False) -> None:
        self._frame = frame
        self.show_depth = show_depth
        self.show_obstacle = show_obstacle
        self.obstacles = []
        self.disp = []
        self.threeD = []
        self.from_frame_get_depth()
        self.from_frame_init()
        self.frame_obstacles_size = len(self.obstacles)
    def from_frame_get_depth(self):
        frame = self._frame
        height, width, _ = frame.shape
        frame = cv2.resize(frame, (1280, 480), interpolation=cv2.INTER_AREA)

        frame1_ = frame[0: height, 0: width // 2]  # 左侧相机
        frame2_ = frame[0: height, width // 2: 1280]  # 右侧相机

        if 1:
            # frame1,frame2 = frame1_,frame2_  #read video
            frame1,frame2 = frame2_,frame1_  #read camera

        else:
            frame2 = cv2.flip(frame2_, -1)  # 旋转180
            frame1 = cv2.flip(frame1_, -1)  # 旋转180
        
        
        img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
        num = 4
        blockSize = 5
        unq = 8
        max_diff = 0
        pre_filtercap = 15
        speckle_winsize = 100
        param = {
            "minDisparity": 0,  # 最小视差值(int类型)，通常情况下为0。此参数决定左图中的像素点在右图匹配搜索的起点。最小视差值越小，视差图右侧的黑色区域越大
            "numDisparities": 16 * num,  # 视差搜索范围，其值必须为16的整数倍且大于0。视差窗口越大，视差图左侧的黑色区域越大
            "blockSize": blockSize,  # 匹配块大小（SADWindowSize(SAD代价计算的窗口大小)）,大于1的奇数。默认为5,一般在3~11之间
            "P1": 8 * 3 * blockSize * blockSize,
            # P1是相邻像素点视差增/减 1 时的惩罚系数；需要指出，在动态规划时，P1和P2都是常数。一般：P1=8*通道数*blockSize*blockSize，P2=4*P1
            "P2": 32 * 3 * blockSize * blockSize,  # P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。p2值越大，差异越平滑
            "disp12MaxDiff": max_diff,  # 左右视差图的最大容许差异（左右一致性检测,超过将被清零），默认为-1，即不执行左右视差检查。
            "preFilterCap": pre_filtercap,  # 图像预处理参数，水平sobel预处理后，映射滤波器大小。默认为15
            "uniquenessRatio": unq,
            # 视差唯一性检测百分比，视差窗口范围内最低代价是次低代价的(1+uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
            "speckleWindowSize": speckle_winsize,
            # 视差连通区域像素点个数的大小。若大于，视差值认为有效，否则认为当前视差值是噪点。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
            "speckleRange": 2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
            # "mode":false  # 模式,取值0,1,2,3。默认被设置为false。
        }
        left_matcher = cv2.StereoSGBM_create(**param)

        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        left_disp = left_matcher.compute(imgL, imgR)
        right_disp = right_matcher.compute(imgR, imgL)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        # sigmaColor典型范围值为0.8-2.0
        wls_filter.setLambda(8000.)
        wls_filter.setSigmaColor(1.3)
        wls_filter.setLRCthresh(24)
        wls_filter.setDepthDiscontinuityRadius(3)

        filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)
        # 将图片扩展至3d空间中，其z方向的值则为当前的距离
        threeD = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16., camera_configs.Q)
        the_max_num =threeD.max()
        filtered_disp[filtered_disp < 0] = 0
        filtered_disp[filtered_disp > 1000] = 1000
        # threeD[threeD>50000] = 50000
        # threeD[threeD<0] = 0
        # print(the_max_num > 500)

        # filtered_disp = weighted_least_squares_filter(left_disp, right_disp)
        disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
        # disp/=255
        self.disp = disp
        self.threeD = threeD
        if self.show_depth:
            cv2.imshow("left", img1_rectified)
            cv2.imshow("right", img2_rectified)
            cv2.imshow("depth", disp)
            left_disp = cv2.normalize(left_disp, left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                      dtype=cv2.CV_8U)
            right_disp = cv2.normalize(right_disp, right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                       dtype=cv2.CV_8U)
            cv2.imshow("left_disp", left_disp)
            cv2.imshow("right_disp", right_disp)


    def from_frame_init(self):
        # gray_frame = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)

        left_crop = 0
        for x in range(self.disp.shape[1]):
            if np.any(self.disp[:, x] > 0):  # 检查该列是否有非黑色像素
                left_crop = x
                break
        pixels = self.disp.reshape((-1, 1))
        # 将数据转换为float类型
        pixels = np.float32(pixels)
        # 定义K均值聚类的停止标准
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        num_clusters = 5
        # 应用K均值聚类
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        # 将标签重新调整为与图像大小一致的形状
        labels = labels.reshape(self.disp.shape)
        kmeans_contours = []
        labels[:,0:left_crop] = 100
        labels[0, left_crop:] = 100  # y == 0 的所有像素
        labels[-1, left_crop:] = 100  # y == 480 的所有像素
        labels[:, left_crop] = 100  # x == 0 的所有像素
        labels[:, -1] = 100  # x == 640 的所有像素
        for i in range(num_clusters):
            # 创建一个二值图像，仅包含当前聚类的区域
            cluster_mask = np.uint8(labels == i)
            # 寻找轮廓\
            cluster_mask = np.uint8(cluster_mask * 255)
            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 计算聚类区域的面积
            area = cv2.countNonZero(cluster_mask)
            if area <= 100: continue
            center = (0,0)
            # if abs(self.threeD[center[1], center[0]][2]) >= 3500:
            #     continue
            if type(contours) == tuple:
              if abs(self.threeD[contours[0][0][0][1], contours[0][0][0][0]][2]) >= 3500:
                # print('111',contours[0][0][0][1], contours[0][0][0][0])
                continue
            else:
              if abs(self.threeD[contours[0][0][1], contours[0][0][0]][2]) >= 3500:
                # print('222',contours[0][0][1], contours[0][0][0])

                continue
                

            kmeans_contours.append([contours, center, area])
        # 过滤轮廓
        for (contours,contour_center,contour_area) in kmeans_contours:
            if contour_area > 172030:
                continue
            if contour_area < 1000:
                continue
            color = np.random.randint(0, 255, (3,)).tolist()
            area = contour_area
            for contours_i in contours:
                if cv2.contourArea(contours_i) < 1000:
                    continue
                moments = cv2.moments(contours_i)
                center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                cx = int(center[0])
                cy = int(center[1])
                self.obstacles.append(obstacle(id=0, area=area, center_xy_img=(cx, cy), center_xyz_world=(0,0,0), contour_img=contours_i,contour=[], angele=0))
        id = 0
        # 单位像素所占的实际长度 mm
        d = abs(self.threeD[0,-1][0] - self.threeD[0,0][0]) / (self.disp.shape[1] - 1)
        for obstacle_i in self.obstacles:
            if self.show_obstacle:
                cv2.drawContours(self.disp, obstacle_i.contour_img, -1, (255, 255, 255), 5)
            obstacle_i.id = id
            id += 1
            obstacle_i.center_xyz_world = self.threeD[obstacle_i.center_xy_img[1], obstacle_i.center_xy_img[0]]
            obstacle_i.angle = np.arctan2(obstacle_i.center_xyz_world[1], obstacle_i.center_xyz_world[0]), np.arctan2(obstacle_i.center_xyz_world[0], obstacle_i.center_xyz_world[1]), np.arctan2(obstacle_i.center_xyz_world[2],np.sqrt(obstacle_i.center_xyz_world[0]**2+ obstacle_i.center_xyz_world[1]**2))
            obstacle_i.area *= d # mm**2
        # 对每个具体的轮廓进行初始化
        for obstacle_i in self.obstacles:
            #进行降采样
            self.downsample_contour(obstacle_i)

        if self.show_obstacle:
            # 把深度图变成rgb的
            self.disp = cv2.cvtColor(self.disp, cv2.COLOR_GRAY2BGR)
            # 把障碍物的轮廓点画在深度图上
            for obstacle_i in self.obstacles:
                for contour_i in obstacle_i.contour_img:
                    cv2.circle(self.disp, (int(contour_i[0]), int(contour_i[1])), 5, (0, 0, 255), -1)
                
                cv2.circle(self.disp,(obstacle_i.center_xy_img[0],obstacle_i.center_xy_img[1]),5,[255,0,255],-1)
                # print(obstacle_i.center_xy_img[0], obstacle_i.center_xy_img[1])
                # print(obstacle_i.center_xy_img[0], obstacle_i.center_xy_img[1])

                # 定义文字内容和位置
                text = 'Dis:'+str("{:.2f}".format(obstacle_i.center_xyz_world[2]))
                text_position = (obstacle_i.center_xy_img[0] + 20, obstacle_i.center_xy_img[1] - 20)  # 文字位置稍微偏移点的位置  
                text_color = (255, 0, 255)     
                text_font = cv2.FONT_HERSHEY_SIMPLEX  # 字体类型  
                text_scale = 0.8            # 字体大小
                text_thickness = 2          # 文字粗细
                
                # 在图像上标注文字  
                cv2.putText(self.disp, text, text_position, text_font, text_scale, text_color, text_thickness)
                
                cv2.imshow("obstacle", self.disp)


    def get_world_coordinates(self,point) -> list:
        x, y = point[0]
        return self.threeD[y][x]  # 获取对应的深度值
    def downsample_contour(self,obstacle):
        contour = obstacle.contour_img
        # 将轮廓点转换为世界坐标
        contour_world = np.array([self.get_world_coordinates(pt)  for pt in contour])
        contours = np.array([pt[0]  for pt in contour])
        if len(contour_world) < 670:
            cnt = 8
        else:
            cnt = 10
        contour_world = contour_world[::math.ceil(len(contour_world) / cnt)]
        obstacle.contour = contour_world
        obstacle.contour_img = contours[::math.ceil(len(contours) / cnt)]

    def k_means_out(self,gray_image):
        # 将灰度图像转换为一维数组
        pixels = gray_image.reshape((-1, 1))

        # 将数据转换为float类型
        pixels = np.float32(pixels)

        # 定义K均值聚类的停止标准
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # 设置聚类数量，默认为4
        num_clusters = 10

        # 应用K均值聚类
        _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 将标签重新调整为与图像大小一致的形状
        labels = labels.reshape(gray_image.shape)
        center_out = []
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
            if self.threeD[center[1], center[0]][2] >= 3500:
                continue
            center_out.append(center)