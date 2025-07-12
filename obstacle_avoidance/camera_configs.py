## 2024，7.19 小样机头壳标定数据 ##

# import cv2
# import numpy as np
#
# left_camera_matrix = np.array([ [611.051094440804,        0, 350.279169055595],
#                                 [       0, 611.420223767171, 246.177289109503],
#                                 [       0,        0,        1] ])
#
# right_camera_matrix = np.array([ [609.559420165812,        0, 329.953998721447],
#                                  [       0, 610.049310978801, 258.4365959467759],
#                                  [       0,        0,        1] ])
#
# left_distortion = np.array([[0.358126507750081, 0.0961838363966326, -0.0111346228037507, -0.00179613305874396
#                                 , -0.308987506128831]])
#
# right_distortion = np.array([[0.349173153651093, 0.150899031296726, -0.0110036632311055, -0.00206441189584858
#                                  , -0.369809934260496]])
#
# # om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
# # R = cv2.Rodrigues(om)[0] # 使用Rodrigues变换将om变换为R
# R = np.array([ [0.999988978361286, -0.00230414602695138,  -0.00409072940170227],
#                [0.00229921440189715, 0.999996624926287, -0.00120985501997447],
#                [0.00409350327782646, 0.00120043622143488,  0.999990901050501] ])
# T = np.array([-96.8530668278492, 0.286306063115955, -2.22043065515438]) # 平移关系向量
#
# size = (640, 480) # 图像尺寸
#
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, size, R, T)
#
# left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)CV_16SC2

# # 2024/9/3标定
# import cv2
# import numpy as np
#
# # 转置后的相机1的内参矩阵
# left_camera_matrix = np.array([ [618.349493097308, 0, 349.094049769163],
#                                 [0, 617.965829062036, 251.597363596584],
#                                 [0, 0, 1] ])
#
# # 转置后的相机2的内参矩阵
# right_camera_matrix = np.array([ [617.677585773323, 0, 327.988924262676],
#                                  [0, 617.414878283450, 263.040162373324],
#                                  [0, 0, 1] ])
#
# # 畸变系数保持不变
# left_distortion = np.array([[0.357990814404999, 0.176300787100131, -0.369638711288173, -0.00700622720259514,
#                              -0.00196073674878715]])
#
# right_distortion = np.array([[0.365216255551014, 0.117067486036404, -0.208334680069350, -0.00618932976789379,
#                               -0.00275603562858836]])
#
# # 转置后的旋转矩阵
# R = np.array([ [0.999971114032829, 0.00494616009877704, 0.00577118707210835],
#                [-0.00494027886918732, 0.999987263321947, -0.00103287878207499],
#                [-0.00577622235017540, 0.00100433767282990, 0.999982813132906] ])
#
# # 转置后的平移向量
# T = np.array([-96.3717151262569, -0.118674922982770, 0.859084810623968]).T  # 1x3变为3x1
#
# size = (640, 480) # 图像尺寸
#
# # 使用转置后的矩阵进行立体校正
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
#                                                                    right_camera_matrix, right_distortion,
#                                                                    size, R, T)
#
# left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
# # 2024/9/4  空气标定
# import cv2
# import numpy as np
#
# # 转置后的相机1的内参矩阵
# left_camera_matrix = np.array([ [ 470.7677  , -0.0095  ,353.6136],
#                                 [ 0 , 470.9821 , 256.7620],
#                                 [0, 0, 1] ])
#
# # 转置后的相机2的内参矩阵
# right_camera_matrix = np.array([ [ 469.7769  ,  0.3521 , 332.4374],
#                                  [ 0 , 469.9185 , 267.6663],
#                                  [0, 0, 1] ])
#
# # 畸变系数保持不变
# left_distortion = np.array([[-0.003136, -0.031730, -0.000441, -0.000870, -0.023263]])
#
# right_distortion = np.array([[0.001952, -0.057221, -0.000241, -0.000523, 0.020193]])
#
# # 转置后的旋转矩阵
# R = np.array([ [ 1.0000 ,   0.0047  ,  0.0039],
#                [ -0.0046 ,   1.0000  , -0.0013],
#                [-0.0039 ,   0.0013  ,  1.0000] ])
#
# # 转置后的平移向量
# T = np.array([ -95.7786 ,  -0.0655  , -0.5410]).T  # 1x3变为3x1
#
# size = (640, 480) # 图像尺寸
#
# # 使用转置后的矩阵进行立体校正
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
#                                                                    right_camera_matrix, right_distortion,
#                                                                    size, R, T)
#
# left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
# right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
#
'''
#xiao jiaodu water
import cv2
import numpy as np

# 新的相机1（左相机）的内参矩阵
left_camera_matrix = np.array([ [631.7923, 0.0516, 354.6903],
                                [0, 632.2608, 255.8443],
                                [0, 0, 1] ])

# 新的相机2（右相机）的内参矩阵
right_camera_matrix = np.array([ [632.4472, 0.5651, 331.5882],
                                 [0, 632.8025, 266.9300],
                                 [0, 0, 1] ])

# 更新的畸变系数
left_distortion = np.array([[0.401483, 0.046164, -0.001786, 0.002799, 0.097887]])
right_distortion = np.array([[0.388810, 0.226003, -0.001580, 0.000709, -0.770150]])

# 新的旋转矩阵
R = np.array([[1.0000, 0.0044, 0.0071],
              [-0.0044, 1.0000, -0.0007],
              [-0.0071, 0.0007, 1.0000]])

# 新的平移向量
T = np.array([-95.9345, -0.1646, 1.2752]).T  # 1x3变为3x1

size = (640, 480)  # 图像尺寸

# 使用更新后的矩阵进行立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                   right_camera_matrix, right_distortion,
                                                                   size, R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
'''

# # 0919标定图正置
# import cv2  
# import numpy as np  
  
# # 左相机的内参矩阵  
# left_camera_matrix = np.array([[339.1139, -0.1258, 359.4544],  
#                                [0, 339.2689, 302.1624],  
#                                [0, 0, 1.0000]])  
  
# # 右相机的内参矩阵  
# right_camera_matrix = np.array([[337.5361, -0.1728, 341.1505],  
#                                 [0, 337.7689, 280.1167],  
#                                 [0, 0, 1.0000]])  
  
# # 左相机畸变系数  
# left_distortion = np.array([-0.333527, 0.138755, 0.000740, 0.000388, -0.030016])  
  
# # 右相机畸变系数  
# right_distortion = np.array([-0.323774, 0.124742, 0.000423, 0.000343, -0.024107])  
  
# # 旋转矩阵  
# R = np.array([[1.0000, 0.0068, -0.0014],  
#               [-0.0068, 0.9998, 0.0212],  
#               [0.0015, -0.0211, 0.9998]])  
  
# # 平移向量  
# T = np.array([-96.5276, 0.8396, -0.6032])  # 注意这里不需要转置  
  
# size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配  
  
# # 使用更新后的矩阵进行立体校正  
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(  
#     left_camera_matrix, left_distortion,  
#     right_camera_matrix, right_distortion,  
#     size, R, T  
# )  
  
# # 计算校正映射  
# left_map1, left_map2 = cv2.initUndistortRectifyMap(  
#     left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2  
# )  
# right_map1, right_map2 = cv2.initUndistortRectifyMap(  
#     right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2  
# )  


# 大角度 0920空气中标定标定图正置 ,包含倾斜

# import cv2  
# import numpy as np  
  
# # 左相机的内参矩阵  
# left_camera_matrix = np.array([  
#     [337.7759, -0.3629, 359.3753],  
#     [0, 337.2240, 294.3447],  
#     [0, 0, 1.0000]  
# ])  
  
# # 右相机的内参矩阵  
# right_camera_matrix = np.array([  
#     [339.5218, 0.6127, 341.7623],  
#     [0, 340.5494, 283.3478],  
#     [0, 0, 1.0000]  
# ])  
  
# # 左相机畸变系数  
# left_distortion = np.array([-0.321095, 0.123865, 0.002685, 0.000386, -0.023545])  
  
# # 右相机畸变系数  
# right_distortion = np.array([-0.315224, 0.112723, -0.000791, -0.000649, -0.019773])  
  
# # 旋转矩阵  
# R = np.array([  
#     [1.0000, 0.0053, 0.0077],  
#     [-0.0054, 0.9999, 0.0099],  
#     [-0.0077, -0.0099, 0.9999]  
# ])  
  
# # 平移向量  
# T = np.array([-99.9209, -5.9070, 2.8822])  # 注意这里不需要转置  
  
# size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配  
  
# # 使用更新后的矩阵进行立体校正  
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(  
#     left_camera_matrix, left_distortion,  
#     right_camera_matrix, right_distortion,  
#     size, R, T  
# )  
  
# # 计算校正映射  
# left_map1, left_map2 = cv2.initUndistortRectifyMap(  
#     left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2  
# )  
# right_map1, right_map2 = cv2.initUndistortRectifyMap(  
#     right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2  
# )  


# 0920水中标定 dajiaodu water
# import cv2  
# import numpy as np
  
# # 左相机的内参矩阵  
# left_camera_matrix = np.array([  
#     [444.4785, -0.1906, 355.4238],  
#     [0, 444.7927, 301.5674],  
#     [0, 0, 1.0000]  
# ])  
  
# # 右相机的内参矩阵  
# right_camera_matrix = np.array([  
#     [444.5667, -0.0803, 340.5284],  
#     [0, 445.1559, 275.8386],  
#     [0, 0, 1.0000]  
# ])  
  
# # 左相机畸变系数  
# left_distortion = np.array([-0.239267, 0.170382, -0.000592, -0.001494, -0.061234])  
  
# # 右相机畸变系数  
# right_distortion = np.array([-0.232834, 0.144903, -0.005291, -0.000552, -0.024972])  
  
# # 旋转矩阵  
# R = np.array([  
#     [0.9999, 0.0065, -0.0079],  
#     [-0.0063, 0.9996, 0.0280],  
#     [0.0081, -0.0280, 0.9996]  
# ])  
  
# # 平移向量  
# T = np.array([-96.2416, 0.5330, 0.5786])  
  
# size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配  
  
# # 使用更新后的矩阵进行立体校正  
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(  
#     left_camera_matrix, left_distortion,  
#     right_camera_matrix, right_distortion,  
#     size, R, T  
# )  
  
# # 计算校正映射  
# left_map1, left_map2 = cv2.initUndistortRectifyMap(  
#     left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2  
# )  
# right_map1, right_map2 = cv2.initUndistortRectifyMap(  
#     right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2  
# )  



#1219 xintoukebiaodinng air
# import cv2
# import numpy as np

# # 左相机的内参矩阵
# left_camera_matrix = np.array([
#     [335.7585, 0.0286, 328.0116],
#     [0, 336.0400, 269.9832],
#     [0, 0, 1.0000]
# ])

# # 右相机的内参矩阵
# right_camera_matrix = np.array([
#     [336.4823, -0.0983, 352.2795],
#     [0, 336.7380, 296.9772],
#     [0, 0, 1.0000]
# ])

# # 左相机畸变系数
# left_distortion = np.array([-0.318880, 0.117342, 0.000023, -0.000374, -0.020850])

# # 右相机畸变系数
# right_distortion = np.array([-0.321977, 0.120434, -0.000150, 0.000195, -0.021786])

# # 旋转矩阵
# R = np.array([
#     [0.9999, 0.0083, -0.0077],
#     [-0.0083, 1.0000, -0.0005],
#     [0.0077, 0.0006, 1.0000]
# ])

# # 平移向量（注意：基线长度是平移向量的第一个元素，但方向可能相反，取决于相机配置）
# T = np.array([-95.6168, 0.5006, -0.3534])

# size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配

# # 使用更新后的矩阵进行立体校正  
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(  
#     left_camera_matrix, left_distortion,  
#     right_camera_matrix, right_distortion,  
#     size, R, T  
# )  
  
# # 计算校正映射  
# left_map1, left_map2 = cv2.initUndistortRectifyMap(  
#     left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2  
# )  
# right_map1, right_map2 = cv2.initUndistortRectifyMap(  
#     right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2  
# )  

#1219 xintoukebiaodinng water
# import cv2
# import numpy as np
# # 左相机的内参矩阵
# left_camera_matrix = np.array([
#     [444.1909, 0.0186, 325.6804],
#     [0, 444.4205, 267.6553],
#     [0, 0, 1.0000]
# ])
 
# # 右相机的内参矩阵
# right_camera_matrix = np.array([
#     [445.8226, -0.0308, 353.1367],
#     [0, 446.1680, 296.4978],
#     [0, 0, 1.0000]
# ])
 
# # 左相机畸变系数
# left_distortion = np.array([-0.233996, 0.145938, -0.000705, -0.000100, -0.029697])
 
# # 右相机畸变系数
# right_distortion = np.array([-0.239169, 0.158411, -0.000297, 0.000019, -0.040576])
 
# # 旋转矩阵
# R = np.array([
#     [0.9999, 0.0084, -0.0145],
#     [-0.0085, 1.0000, -0.0009],
#     [0.0145, 0.0011, 0.9999]
# ])
 
# # 平移向量（基线长度是平移向量的第一个元素，但方向可能相反，取决于相机配置）
# T = np.array([-95.7544, 0.2917, 0.1037])
 
# size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配
 
# # 使用正确的矩阵进行立体校正
# R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
#     left_camera_matrix, left_distortion,
#     right_camera_matrix, right_distortion,
#     size, R, T,
#     alpha=0  # alpha参数控制校正图像的缩放，0表示不缩放
# )
 
# # 计算校正映射
# left_map1, left_map2 = cv2.initUndistortRectifyMap(
#     left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2
# )
# right_map1, right_map2 = cv2.initUndistortRectifyMap(
#     right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2
# )

#1224 xintouke water
# import cv2
# import numpy as np

# # 左相机的内参矩阵
# left_camera_matrix = np.array([
#     [446.9425, 0.2343, 325.7460],
#     [0, 447.0474, 270.0880],
#     [0, 0, 1.0000]
# ])

# # 右相机的内参矩阵
# right_camera_matrix = np.array([
#     [448.1290, 0.2101, 352.6952],
#     [0, 448.3027, 296.5710],
#     [0, 0, 1.0000]
# ])

# # 左相机畸变系数
# left_distortion = np.array([-0.235267, 0.144287, -0.000751, 0.000247, -0.028519])

# # 右相机畸变系数
# right_distortion = np.array([-0.241239, 0.158700, -0.000380, 0.000495, -0.037674])

# # 旋转矩阵
# R = np.array([
#     [0.9999, 0.0079, -0.0149],
#     [-0.0079, 1.0000, 0.0007],
#     [0.0149, -0.0006, 0.9999]
# ])

# # 平移向量
# T = np.array([-95.9218, 0.6063, -0.5965])  # 注意这里不需要转置

# size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配

# # 使用更新后的矩阵进行立体校正
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
#     left_camera_matrix, left_distortion,
#     right_camera_matrix, right_distortion,
#     size, R, T
# )

# # 计算校正映射
# left_map1, left_map2 = cv2.initUndistortRectifyMap(
#     left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2
# )
# right_map1, right_map2 = cv2.initUndistortRectifyMap(
#     right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2
# )


# 1227 7號機新標定參數
import cv2
import numpy as np

# 左相机的内参矩阵
left_camera_matrix = np.array([
    [446.1312, -0.1607, 360.5116],
    [0, 446.5769, 308.2938],
    [0, 0, 1.0000]
])

# 右相机的内参矩阵
right_camera_matrix = np.array([
    [445.7741, 0.1990, 342.0431],
    [0, 446.7158, 283.7390],
    [0, 0, 1.0000]
])

# 左相机畸变系数
left_distortion = np.array([-0.229800, 0.140721, 0.001900, -0.000648, -0.029280])

# 右相机畸变系数
right_distortion = np.array([-0.220908, 0.111177, 0.000023, -0.000142, -0.009207])

# 旋转矩阵
R = np.array([
    [1.0000, -0.0044, -0.0053],
    [0.0044, 0.9999, 0.0098],
    [0.0052, -0.0099, 0.9999]
])

# 平移向量
T = np.array([-95.7117, -0.3257, -0.4032])  # 注意这里不需要转置

size = (640, 480)  # 假设的图像尺寸，应与您的相机捕获的图像尺寸相匹配

# 使用更新后的矩阵进行立体校正
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    left_camera_matrix, left_distortion,
    right_camera_matrix, right_distortion,
    size, R, T, alpha=0  # alpha 参数控制校正后的图像缩放，0 表示无缩放
)

# 计算校正映射
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2
)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2
)


