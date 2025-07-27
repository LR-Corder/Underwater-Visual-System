# Copyright 2025 Beijing Jiaotong University (BJTU). All rights reserved.

import math
from dataclasses import dataclass
import cv2
import numpy as np
import camera_configs

distance_threshold = 1000  # mm


@dataclass
class Obstacle:
    id: int
    area: float
    center_xy_img: tuple
    center_xyz_world: tuple
    contour_img: list
    contour: list
    angle: float


class FrameObstacles:
    def __init__(self, frame, show_depth=False, show_obstacle=False) -> None:
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

        frame1_ = frame[:, :width // 2]  # Left camera
        frame2_ = frame[:, width // 2:]  # Right camera

        img1_rectified = cv2.remap(frame1_, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(frame2_, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
        imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

        param = {
            "minDisparity": 0,
            "numDisparities": 16 * 4,
            "blockSize": 5,
            "P1": 8 * 3 * 5 * 5,
            "P2": 32 * 3 * 5 * 5,
            "disp12MaxDiff": 0,
            "preFilterCap": 15,
            "uniquenessRatio": 8,
            "speckleWindowSize": 100,
            "speckleRange": 2,
        }

        left_matcher = cv2.StereoSGBM_create(**param)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

        left_disp = left_matcher.compute(imgL, imgR)
        right_disp = right_matcher.compute(imgR, imgL)
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(8000.)
        wls_filter.setSigmaColor(1.3)
        wls_filter.setLRCthresh(24)
        wls_filter.setDepthDiscontinuityRadius(3)

        filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)
        threeD = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16., camera_configs.Q)

        filtered_disp[filtered_disp < 0] = 0
        filtered_disp[filtered_disp > 1000] = 1000

        disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        self.disp = disp
        self.threeD = threeD

        if self.show_depth:
            cv2.imshow("left", img1_rectified)
            cv2.imshow("right", img2_rectified)
            cv2.imshow("depth", disp)

    def from_frame_init(self):
        left_crop = 0
        for x in range(self.disp.shape[1]):
            if np.any(self.disp[:, x] > 0):
                left_crop = x
                break

        pixels = self.disp.reshape((-1, 1))
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        num_clusters = 5
        _, labels, _ = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape(self.disp.shape)

        labels[:, :left_crop] = 100
        labels[0, left_crop:] = 100
        labels[-1, left_crop:] = 100
        labels[:, left_crop] = 100
        labels[:, -1] = 100

        kmeans_contours = []
        for i in range(num_clusters):
            cluster_mask = np.uint8(labels == i)
            cluster_mask = np.uint8(cluster_mask * 255)
            contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.countNonZero(cluster_mask)
            if area <= 100:
                continue
            center = (0, 0)
            if type(contours) == tuple:
                if abs(self.threeD[contours[0][0][0][1], contours[0][0][0][0]][2]) >= 3500:
                    continue
            else:
                if abs(self.threeD[contours[0][0][1], contours[0][0][0]][2]) >= 3500:
                    continue

            kmeans_contours.append([contours, center, area])

        for contours, contour_center, contour_area in kmeans_contours:
            if contour_area > 172030 or contour_area < 1000:
                continue

            color = np.random.randint(0, 255, (3,)).tolist()
            for contours_i in contours:
                if cv2.contourArea(contours_i) < 1000:
                    continue
                moments = cv2.moments(contours_i)
                center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
                cx = int(center[0])
                cy = int(center[1])
                self.obstacles.append(Obstacle(id=0, area=area, center_xy_img=(cx, cy), center_xyz_world=(0, 0, 0),
                                               contour_img=contours_i, contour=[], angle=0))

        id = 0
        d = abs(self.threeD[0, -1][0] - self.threeD[0, 0][0]) / (self.disp.shape[1] - 1)
        for obstacle_i in self.obstacles:
            if self.show_obstacle:
                cv2.drawContours(self.disp, [obstacle_i.contour_img], -1, (255, 255, 255), 5)
            obstacle_i.id = id
            id += 1
            obstacle_i.center_xyz_world = self.threeD[obstacle_i.center_xy_img[1], obstacle_i.center_xy_img[0]]
            obstacle_i.angle = math.atan2(obstacle_i.center_xyz_world[1], obstacle_i.center_xyz_world[0]), \
                               math.atan2(obstacle_i.center_xyz_world[0], obstacle_i.center_xyz_world[1]), \
                               math.atan2(obstacle_i.center_xyz_world[2], math.sqrt(obstacle_i.center_xyz_world[0] ** 2 +
                                                                                   obstacle_i.center_xyz_world[1] ** 2))
            obstacle_i.area *= d

        for obstacle_i in self.obstacles:
            self.downsample_contour(obstacle_i)

        if self.show_obstacle:
            self.disp = cv2.cvtColor(self.disp, cv2.COLOR_GRAY2BGR)
            for obstacle_i in self.obstacles:
                for contour_i in obstacle_i.contour_img:
                    cv2.circle(self.disp, (int(contour_i[0][0]), int(contour_i[0][1])), 5, (0, 0, 255), -1)
                cv2.circle(self.disp, (obstacle_i.center_xy_img[0], obstacle_i.center_xy_img[1]), 5, (255, 0, 255), -1)

                text = f'Dis: {obstacle_i.center_xyz_world[2]:.2f}'
                text_position = (obstacle_i.center_xy_img[0] + 20, obstacle_i.center_xy_img[1] - 20)
                text_color = (255, 0, 255)
                text_font = cv2.FONT_HERSHEY_SIMPLEX
                text_scale = 0.8
                text_thickness = 2

                cv2.putText(self.disp, text, text_position, text_font, text_scale, text_color, text_thickness)

            cv2.imshow("obstacle", self.disp)

    def get_world_coordinates(self, point) -> list:
        x, y = point[0]
        return self.th