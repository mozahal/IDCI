import cv2
import numpy as np
import math
from camera_params import size,left_camera_matrix_single,left_distortion_single,right_distortion_single,right_camera_matrix_single

import camera_params

# camera2_camera_matrix = camera_params.left_camera_matrix_l_r
# camera2_distortion = camera_params.left_distortion_l_r
camera2_camera_matrix = camera_params.left_camera_matrix_single
camera2_distortion = camera_params.left_distortion_single


# camera1_camera_matrix = camera_params.right_camera_matrix_l_r
# camera1_distortion = camera_params.right_distortion_l_r
camera1_camera_matrix = camera_params.right_camera_matrix_single
camera1_distortion = camera_params.right_distortion_single

R = np.matrix([
    [  0.999973946439507,0.00186407713914413,0.00697364027016308],
    [  -0.00188987472900531,0.999991389331520,0.00369454142280429],
    [  -0.00696669331225274,-0.00370762447336160,0.999968858867644],
])

T = np.array([121.425503984653,0.551065597063646,-2.46338377031500])

size = camera_params.size # 图像尺寸
# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(camera1_camera_matrix, camera1_distortion,
                                                                  camera2_camera_matrix, camera2_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(camera2_camera_matrix, camera2_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(camera1_camera_matrix, camera1_distortion, R2, P2, size, cv2.CV_16SC2)



baseline=math.sqrt(math.pow(T[0],2)+math.pow(T[1],2)+math.pow(T[2],2))
# fr=math.sqrt(math.pow(camera1_camera_matrix[0,0],2)/2+math.pow(camera1_camera_matrix[1,1],2)/2)
# fl=math.sqrt(math.pow(camera2_camera_matrix[0,0],2)/2+math.pow(camera2_camera_matrix[1,1],2)/2)
fr=camera1_camera_matrix[0,0]
fl=camera2_camera_matrix[0,0]