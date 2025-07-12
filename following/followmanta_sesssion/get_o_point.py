import cv2
import numpy as np


def contours_find(src):

    # 在边缘添加矩形边
    cv2.rectangle(src, (0, 0), (src.shape[1], src.shape[0]), (255, 255, 0), 30, 4, 0)

    # 将bgr图片转换成灰度图
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 对图像进行边缘提取
    edges = cv2.Canny(gray, 20, 40, apertureSize=3)

    # 第一次找到轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dst = np.zeros_like(src)

    for contour in contours:
        cv2.drawContours(dst, [contour], -1, (255, 255, 255), 5)
    # 对画出来的轮廓图转成灰度图
    dst_gray = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    # 第二次找到轮廓
    contours, _ = cv2.findContours(dst_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dst = np.zeros_like(dst)

    for contour in contours:
        color = np.random.randint(0, 255, (3,)).tolist()
        cv2.drawContours(dst, [contour], -1, color, 5)
    return dst,contours


def Bundle_sort(pa):
    count_Pointsize = len(pa)

    for count_area1 in range(count_Pointsize - 1):
        flag_area = False
        for count_area2 in range(count_Pointsize - 1 - count_area1):
            if type(pa[count_area2+1])==int:
                break
            if pa[count_area2][2] > pa[count_area2 + 1][2]:
                pa[count_area2],pa[count_area2+1] =pa[count_area2+1],pa[count_area2]
                flag_area = True

        if not flag_area:
            break




# 定义Point3d类
class Point3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

def compute_center_of_gravity(dst2, contours, contours_size, pa,depth):
    count_Pointsize = 0
    pt =[0] * 100
    for i in range(contours_size):
        if cv2.contourArea(contours[i]) > 1000:
            temp = np.array(contours[i])
            leftmost = tuple(temp[temp[:, :, 0].argmin()][0])
            rightmost = tuple(temp[temp[:, :, 0].argmax()][0])
            topmost = tuple(temp[temp[:, :, 1].argmin()][0])
            bottommost = tuple(temp[temp[:, :, 1].argmax()][0])
            color = (0, 255, 255)
            moment = cv2.moments(temp, False)
            if moment['m00'] != 0:
                x = int(moment['m10'] / moment['m00'])  # 计算重心横坐标
                y = int(moment['m01'] / moment['m00'])  # 计算重心纵坐标

                x_rect, y_rect, width, height = cv2.boundingRect(temp)


                p = (x, y)  # 重心坐标
                cv2.circle(dst2, p, 1, color, 30, 8)  # 原图画出重心坐标
                # pa[count_Pointsize] = (x, y, int(cv2.contourArea(contours[i], False)))
                pa[count_Pointsize] = (x, y, int(cv2.contourArea(contours[i], False)), leftmost,rightmost,topmost,bottommost)
                count_Pointsize += 1

    return count_Pointsize

def nms_point(White, pa, pb, count_Pointsize,threeD):
    result_target = []
    result_object = []
    index_size = count_Pointsize
    index_result = 0
    threshold_set = 50
    p = []

    while index_size > 0:
        pb[index_result] = pa[index_size - 1]
        leftmost = pb[index_result][3]
        rightmost = pb[index_result][4]
        topmost = pb[index_result][5]
        bottommost = pb[index_result][6]
        p.append((pb[index_result][0], pb[index_result][1],leftmost ,rightmost ,topmost ,bottommost ))




        count_point = 0
        while count_point < index_size - 1:
            distance = np.sqrt(
                (pb[index_result][0] - pa[count_point][0]) ** 2 + (pb[index_result][1] - pa[count_point][1]) ** 2)

            if distance < threshold_set:
                for count_point2 in range(count_point, index_size - 1):
                    pa[count_point2] = pa[count_point2 + 1]
                index_size -= 1
            else:
                count_point += 1

        index_size -= 1
        index_result += 1

    for nms_point in range(index_result):
        y =p[nms_point][1]
        x =p[nms_point][0]
        area = pb[nms_point][2]
        chang =abs(threeD[ p[nms_point][2][1]][p[nms_point][2][1]]-threeD[p[nms_point][3][1]][p[nms_point][3][1]])
        kuan = abs(threeD[p[nms_point][4][1]][p[nms_point][4][1]]-threeD[p[nms_point][5][1]][p[nms_point][5][1]])
        # ----------------设置距离--------------------------
        if abs(threeD[y][x][2])<8:
            cv2.circle(White, (p[nms_point][0], p[nms_point][1]), 1, (0, 0, 255), 30, 8)
            # print("target:",p[nms_point][0], p[nms_point][1],area,chang,kuan)
            result_target.append([p[nms_point][0], p[nms_point][1],area,chang,kuan])

        else:
            cv2.circle(White, (p[nms_point][0], p[nms_point][1]), 1, (0, 255, 255), 30, 8)
            result_object.append([p[nms_point][0], p[nms_point][1],area,chang,kuan])

    cv2.imshow("final_result", White)
    # for i in result_object:
    #

    return result_target,result_object

def result_point(src,threeD):  #depth ---> [point]   point = (x,y)


    pa = [0]*100
    pb = []
    contours = []
    if src is None:
        print("can not load this pic!")
        exit(0)

    White = src
    dst,contours = contours_find(src)

    count_Pointsize = compute_center_of_gravity(dst, contours,len(contours),pa,src)

    Bundle_sort(pa)
    pb = [Point3d(0, 0, 0) for _ in range(100)]
    result_target,result_object = nms_point(White, pa, pb, count_Pointsize,threeD)
    return result_target
