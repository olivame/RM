import cv2
import numpy as np

img = cv2.imread('/home/ldk/yolov5_camera_final_1/z_pics/9.bmp')
P = [[1922.3,0,1580],
     [0, 1923, 1023.6],
     [0, 0, 1]]
K = [-0.4551, 0.2824,-0.0017, 0.0026,-0.1042]
img_distort = cv2.undistort(img, np.array(P), np.array(K))
#img_diff = cv2.absdiff(img, img_distort)
#cv2.namedWindow("img",cv2.WINDOW_NORMAL)
#cv2.imshow('img', img)
cv2.namedWindow("img_distort",cv2.WINDOW_NORMAL)
cv2.imshow('img_distort', img_distort)
#cv2.imshow('img_absdiff', img_diff)
#cv2.imwrite('/home/ldk/yolov5_camera_final_1/z_pics/distorted_res.png', img_distort)
cv2.waitKey(0)
