# # 单目相机2024.11.05，完好的单目相机
# import cv2  
# import numpy as np  
  
# # 左相机内参矩阵 (根据 cameraParams.IntrinsicMatrix)  
# left_camera_matrix = np.array([  
#     [377.4276236643150, 0.0031288096633, 319.6520423980789],  
#     [0, 377.6352854802945, 218.2777212015161],  
#     [0, 0, 1]
# ])  
  
# # 左相机畸变系数 (根据 cameraParams.RadialDistortion 和 cameraParams.TangentialDistortion)  
# # 注意：这里我们将径向畸变和切向畸变合并成一个数组  
# left_distortion = np.array([-0.001920627940025, -0.015728408450596, -0.000372018458586,
#                             -0.000157976298467, 0.003219191673555]) 
# # 注意：由于cameraParams只给出了两个切向畸变系数，并且它们是以1e-3为单位的，  
# # 我们已经将它们转换为了与径向畸变系数相同的单位。  
# # 同时，我们假设没有提供额外的径向畸变系数（如k4, k5等），因此不使用它们。  
  
# # 右相机内参矩阵和畸变系数（这些看起来是已经给出的，并且假设是正确的）  
# right_camera_matrix = np.array([  
#     [465.4012, -0.8540, 338.8338],  
#     [0, 465.3984, 298.6513],  
#     [0, 0, 1]  
# ])  
# right_distortion = np.array([  
#     0.010200, -0.077743, -0.000413, -0.000563, 0.032876  
# ])  
  
# # 旋转和平移关系（这些看起来也是已经给出的，并且假设是正确的）  
# # om = np.array([0.01911, 0.03125, -0.00960], dtype=np.float32)  # 旋转关系向量  
# # 如果需要使用Rodrigues变换来计算旋转矩阵R，可以取消以下两行的注释  
# # R, _ = cv2.Rodrigues(om)  
# # 但由于您已经给出了R，我们将直接使用它  
# R = np.array([  
#     [0.9999, -0.0123, -0.0059],  
#     [0.0124, 0.9998, 0.0151],  
#     [0.0057, -0.0152, 0.9999]  
# ])  
# T = np.array([[-171.3975], [-0.4235], [1.9587]])  # 平移关系向量  
  
# # 图像尺寸  
# size = (640, 480)  
  
# # 进行立体更正  
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(  
#     left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, size, R, T  
# )  
  
# # 计算更正 map  
# left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)  
# right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)



import cv2  
import numpy as np  

# 左相机内参矩阵 (基于 cameraParams.IntrinsicMatrix)
left_camera_matrix = np.array([  
    [377.4276236643150, 0.0031288096633, 319.6520423980789],  
    [0, 377.6352854802945, 218.2777212015161],  
    [0, 0, 1]
])  

# 左相机畸变系数 (基于 cameraParams.RadialDistortion 和 cameraParams.TangentialDistortion)
# 包括径向畸变和切向畸变，单位已转换为1e-0级别
left_distortion = np.array([
    -0.001920627940025, -0.015728408450596, -0.000372018458586,
    -0.000157976298467, 0.003219191673555
])  

# 右相机内参矩阵和畸变系数（假设已知）
right_camera_matrix = np.array([  
    [465.4012, -0.8540, 338.8338],  
    [0, 465.3984, 298.6513],  
    [0, 0, 1]  
])  
right_distortion = np.array([
    0.010200, -0.077743, -0.000413, -0.000563, 0.032876  
])  

# 左右相机的旋转和平移关系 (假设已知)
R = np.array([  
    [0.9999, -0.0123, -0.0059],  
    [0.0124, 0.9998, 0.0151],  
    [0.0057, -0.0152, 0.9999]  
])  
T = np.array([[-171.3975], [-0.4235], [1.9587]])  

# 图像尺寸 (根据实际图像尺寸设定)
size = (640, 480)  

# 进行立体更正 (stereo rectification)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(  
    left_camera_matrix, left_distortion, 
    right_camera_matrix, right_distortion, 
    size, R, T  
)  

# 生成立体校正映射 (mapping) 表，用于畸变校正和图像对齐
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2
)  
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2
)