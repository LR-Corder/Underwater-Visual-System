import numpy as np
import cv2
import  time
import camera_configs


# camera = cv2.VideoCapture("C:/Users/h/Documents/WeChat Files/wxid_95wh5l7ndw1f21/FileStorage/File/2024-03/stereo_video3.avi")
camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(r"C:\Users\h\Downloads\track_in01.mp4")

print(cv2.__version__)
last_dis = -125
# 添加点击事件，打印当前点的距离
def callbackFunc(e, origin_x, origin_y, f, p) :

    if e == cv2.EVENT_LBUTTONDOWN:

        y1 = origin_y - 10
        x1 = origin_x - 10
        y2 = origin_y + 10
        x2 = origin_x - 10
        y3 = origin_y - 10
        x3 = origin_x + 10
        y4 = origin_y + 10
        x4 = origin_x + 10
        y5 = origin_y - 5
        x5 = origin_x - 5
        y6 = origin_y + 5
        x6 = origin_x - 5
        y7 = origin_y - 5
        x7 = origin_x + 5
        y8 = origin_y + 5
        x8 = origin_x + 5
        points = [(y1, x1), (y2, x2), (y3, x3), (y4, x4), (y5, x5), (y6, x6), (y7, x7), (y8, x8)]
        min_distance = threeD[int(origin_y)][int(origin_x)][2]
        min_point = (origin_y, origin_x)
        for y, x in points:
            # 计算当前点的距离
            distance = threeD[int(y)][int(x)][2]
            # 如果当前点的距离小于最小距离，则更新最小距离和最小点
            if abs(distance) < abs(min_distance):
                min_distance = distance
                min_point = (y, x)
        dis = abs(threeD[min_point[0]][min_point[1]][2])
        if dis > 10000 or (abs(dis - last_dis) > 500 and last_dis != -125): return
        print(dis)

# cv2.setMouseCallback("depth", callbackFunc, None)

while True:
    start = time.time()
    ret1, frame = camera.read()
    # ret2, frame2 = camera2.read()
    height, width, _ = frame.shape
    frame2 =frame[:height, :width//2]
    frame1 = frame[:height, width//2:]


    if not ret1 :
        break


    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
    # cv2.imshow("left", img1_rectified)
    # cv2.imshow("1",img1_rectified)
    # cv2.waitKey(0)
    # img1_rectified = cv2.flip(img1_rectified , 0)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
    # img2_rectified = cv2.flip(img2_rectified, 0)
    # 将图片置为灰度图，为StereoBM作准备
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    # imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 两个trackbar用来调节不同的参数查看效果
    num = 4
    # if num == 0:
    #     num += 1
    blockSize = 5
    # if blockSize % 2 == 0:
    #     blockSize += 1
    # if blockSize < 5:
    #     blockSize = 5
    unq = 5
    max_diff = 0
    pre_filtercap = 15
    speckle_winsize = 100
    # num = 7
    # blockSize = 13

    # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
    # stereo = cv2.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    param = {
        "minDisparity":0,  # 最小视差值(int类型)，通常情况下为0。此参数决定左图中的像素点在右图匹配搜索的起点。最小视差值越小，视差图右侧的黑色区域越大
        "numDisparities":16 * num,  # 视差搜索范围，其值必须为16的整数倍且大于0。视差窗口越大，视差图左侧的黑色区域越大
        "blockSize":blockSize,  # 匹配块大小（SADWindowSize(SAD代价计算的窗口大小)）,大于1的奇数。默认为5,一般在3~11之间
        "P1":8 * 3 * blockSize * blockSize,
        # P1是相邻像素点视差增/减 1 时的惩罚系数；需要指出，在动态规划时，P1和P2都是常数。一般：P1=8*通道数*blockSize*blockSize，P2=4*P1
        "P2":32 * 3 * blockSize * blockSize,  # P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。p2值越大，差异越平滑
        "disp12MaxDiff":max_diff,  # 左右视差图的最大容许差异（左右一致性检测,超过将被清零），默认为-1，即不执行左右视差检查。
        "preFilterCap":pre_filtercap,  # 图像预处理参数，水平sobel预处理后，映射滤波器大小。默认为15
        "uniquenessRatio":unq,
        # 视差唯一性检测百分比，视差窗口范围内最低代价是次低代价的(1+uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
        "speckleWindowSize":speckle_winsize,  # 视差连通区域像素点个数的大小。若大于，视差值认为有效，否则认为当前视差值是噪点。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
        "speckleRange":2  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
        # "mode":false  # 模式,取值0,1,2,3。默认被设置为false。
    }
    # stereo = cv2.StereoSGBM_create(
    #     minDisparity=0,  # 最小视差值(int类型)，通常情况下为0。此参数决定左图中的像素点在右图匹配搜索的起点。最小视差值越小，视差图右侧的黑色区域越大
    #     numDisparities=16 * num,  # 视差搜索范围，其值必须为16的整数倍且大于0。视差窗口越大，视差图左侧的黑色区域越大
    #     blockSize=blockSize,  # 匹配块大小（SADWindowSize(SAD代价计算的窗口大小)）,大于1的奇数。默认为5,一般在3~11之间
    #     P1=8 * 3 * blockSize * blockSize,
    #     # P1是相邻像素点视差增/减 1 时的惩罚系数；需要指出，在动态规划时，P1和P2都是常数。一般：P1=8*通道数*blockSize*blockSize，P2=4*P1
    #     P2=32 * 3 * blockSize * blockSize,  # P2是相邻像素点视差变化值大于1时的惩罚系数。P2必须大于P1。p2值越大，差异越平滑
    #     disp12MaxDiff=20,  # 左右视差图的最大容许差异（左右一致性检测,超过将被清零），默认为-1，即不执行左右视差检查。
    #     preFilterCap=15,  # 图像预处理参数，水平sobel预处理后，映射滤波器大小。默认为15
    #     uniquenessRatio=0,
    #     # 视差唯一性检测百分比，视差窗口范围内最低代价是次低代价的(1+uniquenessRatio/100)倍时，最低代价对应的视差值才是该像素点的视差，否则该像素点的视差为 0，通常为5~15.
    #     speckleWindowSize=100,  # 视差连通区域像素点个数的大小。若大于，视差值认为有效，否则认为当前视差值是噪点。将其设置为0可禁用斑点过滤。否则，将其设置在50-200的范围内。
    #     speckleRange=2,  # 视差变化阈值，每个连接组件内的最大视差变化。如果你做斑点过滤，将参数设置为正值，它将被隐式乘以16.通常，1或2就足够好了
    #     mode= 2 # 模式,取值0,1,2,3。默认被设置为false。
    # )
    # cv2.STEREO_SGBM_MODE_SGBM_3WAY
    # stereo = cv2.StereoSGBM_create(**param)
    # disparity = stereo.compute(imgL, imgR)
    #
    # disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.medianBlur(disp, ksize=3, dst=disp)
    # threeD = cv2.reprojectImageTo3D(disp.astype(np.float32) / 16., camera_configs.Q)

    ## 接上面的参数
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

    disp = cv2.normalize(filtered_disp, filtered_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将图片扩展至3d空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(filtered_disp.astype(np.float32) / 16., camera_configs.Q)
    # cv2.medianBlur(disp, ksize=3, dst=disp)


    # ret, thresh = cv2.threshold(disp, 127, 255, 0)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # print(contours)
    # approx_contours = []
    #
    # #减少轮廓的数据量
    # for contour in contours:
    #     epsilon = epsilon = 0.01 * cv2.arcLength(contour, True)
    #     approx = cv2.approxPolyDP(contour, epsilon, True)
    #     approx_contours.append(approx)
        #shape = (3, 12, 1 ,2) id,count, 1,2(x,y)

    # disp = cv2.drawContours(disp,  approx_contours, -1, (0, 255, 0), 3)
    # disp_app = cv2.drawContours(disp, approx_contours, -1, (0, 255, 0), 3)

    cv2.imshow("left", frame1)
    cv2.imshow("right", frame2)
    cv2.imshow("depth", disp)
    # print(time.time()-start)
    # cv2.imshow("disp_app", disp_app)
    # print(len(approx_contours))
    # print(len(approx_contours[0]))
    # print(len(approx_contours[0][0]))
    # for i in range(len(contours)):
    #     for x,y in contours[0]:
    #         print(threeD[y][x])
    # input("按下 Enter 键继续...")
    key = cv2.waitKey(50)
    if key == ord("q"):
        break
    elif key == ord("s"):
        cv2.imwrite("./snapshot/BM_left.jpg", imgL)
        cv2.imwrite("./snapshot/BM_right.jpg", imgR)
        cv2.imwrite("./snapshot/BM_depth.jpg", disp)

camera.release()

cv2.destroyAllWindows()
