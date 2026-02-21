import cv2
import numpy as np
import math
from camera_params import left_camera_matrix_l_m,left_distortion_l_m,ray_camera_matrix_l_m,ray_distortion_l_m,size_ray

camera2_camera_matrix = left_camera_matrix_l_m
camera2_distortion = left_distortion_l_m



camera1_camera_matrix = ray_camera_matrix_l_m
camera1_distortion = ray_distortion_l_m
size=size_ray
R = np.matrix([
    [  0.99994657,-0.00765551,-0.0069461],
    [  0.00767959, 0.99996457, 0.00344698],
    [  0.00691946,-0.00350014, 0.99996993],
])

T = np.array([59.34422316,2.24425854,-7.05888369])


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
