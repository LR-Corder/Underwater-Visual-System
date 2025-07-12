# 12.04 after
from datetime import datetime

import cv2
import math
import numpy as np
import camera_configs

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
# 读取标定

# calibration_src = cv2.imread(r"C:\Users\h\Desktop\calibration_img3.png")
# 修改像素为640 * 480
# calibration_src = cv2.resize(calibration_src, (640, 480))
# 参数
marker_size_B = 500
marker_size_L = 45

# 获取ArUco字典
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_250)

# 左相机的内参矩阵
camera_matrix_left = camera_configs.left_camera_matrix

# 左相机畸变系数
dist_coeff_left = camera_configs.left_distortion
calibration_source = cv2.VideoCapture(5)

while True:
    ret, calibration_src = calibration_source.read()
    cv2.namedWindow("src",0)
    cv2.resizeWindow("src", 640, 480)
    cv2.imshow("src", calibration_src)
    cv2.waitKey(1)
    corners, ids, _ = cv2.aruco.detectMarkers(calibration_src, dictionary)
    if len(ids) <= 0: continue 
    print(f"Please judge the numbers of the aruco marker: {len(ids)} .")
    # 获得基准
    if 0 in ids:
        index = np.where( ids == 0)[0][0]
    else:
        continue
        # raise ValueError("No 0 aruco marker.")
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[index], marker_size_B,
                                                                camera_matrix_left,
                                                                dist_coeff_left)
    aruco_x =  round(float(tvec[0][0][0]), 1)
    aruco_y =  round(float(tvec[0][0][1]), 1)
    aruco_tvec = tvec
    aruco_rvec = rvec
    aruco_trans = np.eye(4)
    aruco_trans[:3, :3], _ = cv2.Rodrigues(rvec)
    aruco_trans[:3, 3] = aruco_tvec.flatten()
    print(f"trans_star1: {aruco_trans}")
    output_msg = "-----------------output----------------- \n"
    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
    
            aruco_info = {'Id': int(ids[i].item())}
            print(aruco_info['Id'])
            # 设置标记大小
            if aruco_info['Id'] == 0:
                marker_size = marker_size_B
            else:
                marker_size = marker_size_L

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size,
                                                                camera_matrix_left,
                                                                dist_coeff_left)


            if aruco_info['Id'] == 0:  # aruco_info['Id'] == 0:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]), 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]), 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                output_msg += f'''
                \n
                0：
                aruco_0 ={aruco_trans}
                \n
                '''
            elif aruco_info['Id'] == 48:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) + 259.4, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]) + 255.3, 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2= np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                48：
                aruco_48 = { np.linalg.inv(aruco_trans_star_2) @ aruco_trans }
                \n
                '''
            elif aruco_info['Id'] == 47:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) - 250.1, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]), 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2 = np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                47：
                aruco_47 = { np.linalg.inv(aruco_trans_star_2)  @ aruco_trans}
                \n
                '''
            elif aruco_info['Id'] == 11:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) + 2.3, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]), 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2 = np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                11：
                aruco_11 = { np.linalg.inv(aruco_trans_star_2)  @ aruco_trans}
                \n
                '''
            elif aruco_info['Id'] == 5:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) - 42, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]) + 2, 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2 = np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                5：
                aruco_5 = {np.linalg.inv(aruco_trans_star_2)  @ aruco_trans}
                \n
                '''
            elif aruco_info['Id'] == 46:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) - 89.6, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]) + 49, 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2 = np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                46：
                aruco_46 = {np.linalg.inv(aruco_trans_star_2)  @ aruco_trans}
                \n
                '''
            elif aruco_info['Id'] == 45:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) + 262, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]) - 220, 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2 = np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                45：
                aruco_45 = {np.linalg.inv(aruco_trans_star_2)  @ aruco_trans}
                \n
                '''
            else:
                # 更新数据框
                aruco_info[f'Position_X'] = round(float(tvec[0][0][0]) - 241.3, 1)
                aruco_info[f'Position_Y'] = round(float(tvec[0][0][1]) - 230.6, 1)
                aruco_info[f'Position_Z'] = round(float(tvec[0][0][2]), 1)
                aruco_info[f'Rotation_X'] = round(rotationVectorToEulerAngles(rvec)[0], 2)
                aruco_info[f'Rotation_Y'] = round(rotationVectorToEulerAngles(rvec)[1], 2)
                aruco_info[f'Rotation_Z'] = round(rotationVectorToEulerAngles(rvec)[2], 2)
                aruco_trans_star_2 = np.eye(4)
                aruco_trans_star_2[:3, :3], _ = cv2.Rodrigues(rvec)
                aruco_trans_star_2[:3, 3] = tvec.flatten()
                output_msg += f'''
                \n
                else：
                aruco_else = {np.linalg.inv(aruco_trans_star_2)  @ aruco_trans}
                \n
                '''

    print(output_msg)




