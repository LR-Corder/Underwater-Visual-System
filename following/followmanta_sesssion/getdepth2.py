import numpy as np
import cv2
import time
import camera_configs


def get_depth(frame1,frame2,x,y):
    start = time.time()
    img1_rectified = cv2.remap(imgL, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 根据更正map对图片进行重构
   

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    blockSize = 13
    img_channels = 3
    param = {
        "minDisparity": 0,  # 最小视差值(int类型)，通常情况下为0。此参数决定左图中的像素点在右图匹配搜索的起点。最小视差值越小，视差图右侧的黑色区域越大
        "numDisparities": 16*4,  # 视差搜索范围，其值必须为16的整数倍且大于0。视差窗口越大，视差图左侧的黑色区域越大
        "blockSize": blockSize,  # 匹配块大小（SADWindowSize(SAD代价计算的窗口大小)）,大于1的奇数。默认为5,一般在3~11之间
        "P1": 8 * img_channels * blockSize * blockSize,
        # P1是相邻像素点视差增/减 1 时的惩罚系数；需要指出，在动态规划时，P1和P2都是常数。一般：P1=8*通道数*blockSize*blockSize，P2=4*P1
        "P2": 32 * img_channels * blockSize * blockSize,  # P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。p2值越大，差异越平滑
        "disp12MaxDiff": 20,  # 左右视差图的最大容许差异（左右一致性检测,超过将被清零），默认为-1，即不执行左右视差检查。
        "preFilterCap": 15,  # 图像预处理参数，水平sobel预处理后，映射滤波器大小。默认为15
        "uniquenessRatio": 10,
        # 视差唯一性检测百分比，视差窗口范围内最低代价是次低代价的(1+uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
        "speckleWindowSize": 100,  # 视差连通区域像素点个数的大小。若大于，视差值认为有效，否则认为当前视差值是噪点。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
        "speckleRange": 2,  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
        "mode": cv2.STEREO_SGBM_MODE_HH  # 模式,取值0,1,2,3。默认被设置为false。
    }
    left_matcher = cv2.StereoSGBM_create(**param)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    left_disp = left_matcher.compute(imgL, imgR)
    disp = cv2.normalize(left_disp, left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imshow("disp",disp)
    cv2.waitKey(10)
    right_disp = right_matcher.compute(imgR, imgL)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    # # sigmaColor典型范围值为0.8-2.0
    wls_filter.setLambda(8000.)
    wls_filter.setSigmaColor(1.3)
    wls_filter.setLRCthresh(24)
    wls_filter.setDepthDiscontinuityRadius(3)
    filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)
    disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16., camera_configs.Q)
    end = time.time() - start
    # return threeD[int(y)][int(x)][0], threeD[int(y)][int(x)][1], threeD[int(y)][int(x)][2], threeD
    return threeD
import numpy as np
import cv2
import time
import camera_configs


def get_depthmap(frame1,frame2,x,y):
    start = time.time()
    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(imgL, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    blockSize = 13
    img_channels = 3
    param = {
        "minDisparity": 0,  # 最小视差值(int类型)，通常情况下为0。此参数决定左图中的像素点在右图匹配搜索的起点。最小视差值越小，视差图右侧的黑色区域越大
        "numDisparities": 16*4,  # 视差搜索范围，其值必须为16的整数倍且大于0。视差窗口越大，视差图左侧的黑色区域越大
        "blockSize": blockSize,  # 匹配块大小（SADWindowSize(SAD代价计算的窗口大小)）,大于1的奇数。默认为5,一般在3~11之间
        "P1": 8 * img_channels * blockSize * blockSize,
        # P1是相邻像素点视差增/减 1 时的惩罚系数；需要指出，在动态规划时，P1和P2都是常数。一般：P1=8*通道数*blockSize*blockSize，P2=4*P1
        "P2": 32 * img_channels * blockSize * blockSize,  # P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。p2值越大，差异越平滑
        "disp12MaxDiff": 20,  # 左右视差图的最大容许差异（左右一致性检测,超过将被清零），默认为-1，即不执行左右视差检查。
        "preFilterCap": 15,  # 图像预处理参数，水平sobel预处理后，映射滤波器大小。默认为15
        "uniquenessRatio": 10,
        # 视差唯一性检测百分比，视差窗口范围内最低代价是次低代价的(1+uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
        "speckleWindowSize": 100,  # 视差连通区域像素点个数的大小。若大于，视差值认为有效，否则认为当前视差值是噪点。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
        "speckleRange": 2,  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
        "mode": cv2.STEREO_SGBM_MODE_HH  # 模式,取值0,1,2,3。默认被设置为false。
    }
    left_matcher = cv2.StereoSGBM_create(**param)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    left_disp = left_matcher.compute(imgL, imgR)
    right_disp = right_matcher.compute(imgR, imgL)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
    # # sigmaColor典型范围值为0.8-2.0
    wls_filter.setLambda(8000.)
    wls_filter.setSigmaColor(1.3)
    wls_filter.setLRCthresh(24)
    wls_filter.setDepthDiscontinuityRadius(3)
    filtered_disp = wls_filter.filter(left_disp, imgL, disparity_map_right=right_disp)
    disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16., camera_configs.Q)
    end = time.time() - start
    # return threeD[int(y)][int(x)][0], threeD[int(y)][int(x)][1], threeD[int(y)][int(x)][2], threeD
    return disp