# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 23:01:59 2022

@author: lenovo
"""

import cv2
import numpy as np

ROBOT_BLOODS=[100,50,10,30,50,10,1500,10,100,50,10,30,50,10,60,5000]

newImageInfo=(540,960,3) #定义图片的宽高信息
img=np.zeros(newImageInfo,np.uint8)

tag_pixel_x1=750
tag_pixel_x2=840
ROBOT_red=ROBOT_BLOODS[0:8]
ROBOT_blue=ROBOT_BLOODS[8:16]
for i in range(0,8):
    robot_id=i+1
    tag_pixel_y=200+i*20
    if i<5:
        blood_percent_r=ROBOT_red[i]/500
        blood_percent_b=ROBOT_blue[i]/500
        #红方
        robot_tag_r='R'+str(robot_id)
        cv2.putText(img, robot_tag_r, (tag_pixel_x1,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)        
        cv2.line(img,(tag_pixel_x1+30,tag_pixel_y-5),(int(tag_pixel_x1+30+50*blood_percent_r),tag_pixel_y-5),(0,0,255),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_red[i]), (tag_pixel_x1+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
        #蓝方
        robot_tag_b='B'+str(robot_id)
        cv2.putText(img, robot_tag_b, (tag_pixel_x2,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.line(img,(tag_pixel_x2+30,tag_pixel_y-5),(int(tag_pixel_x2+30+50*blood_percent_b),tag_pixel_y-5),(255,0,0),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_blue[i]), (tag_pixel_x2+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)  
    elif i==5:      #哨兵
        blood_percent_r=ROBOT_red[i]/600
        blood_percent_b=ROBOT_blue[i]/600
        #红方
        robot_tag_r='SB'
        cv2.putText(img, robot_tag_r, (tag_pixel_x1,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.line(img,(tag_pixel_x1+30,tag_pixel_y-5),(int(tag_pixel_x1+30+50*blood_percent_r),tag_pixel_y-5),(0,0,255),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_red[i]), (tag_pixel_x1+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
        #蓝方
        robot_tag_b='SB'
        cv2.putText(img, robot_tag_b, (tag_pixel_x2,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.line(img,(tag_pixel_x2+30,tag_pixel_y-5),(int(tag_pixel_x2+30+50*blood_percent_b),tag_pixel_y-5),(255,0,0),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_blue[i]), (tag_pixel_x2+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
    elif i==6:      #前哨站
        blood_percent_r=ROBOT_red[i]/1500
        blood_percent_b=ROBOT_blue[i]/1500
        #红方
        robot_tag_r='PO'
        cv2.putText(img, robot_tag_r, (tag_pixel_x1,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.line(img,(tag_pixel_x1+30,tag_pixel_y-5),(int(tag_pixel_x1+30+50*blood_percent_r),tag_pixel_y-5),(0,0,255),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_red[i]), (tag_pixel_x1+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
        #蓝方
        robot_tag_b='PO'
        cv2.putText(img, robot_tag_b, (tag_pixel_x2,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.line(img,(tag_pixel_x2+30,tag_pixel_y-5),(int(tag_pixel_x2+30+50*blood_percent_b),tag_pixel_y-5),(255,0,0),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_blue[i]), (tag_pixel_x2+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
    elif i==7:      #基地
        blood_percent_r=ROBOT_red[i]/5000
        blood_percent_b=ROBOT_blue[i]/5000
        #红方
        robot_tag_r='BS'
        cv2.line(img,(tag_pixel_x1+30,tag_pixel_y-5),(int(tag_pixel_x1+30+50*blood_percent_r),tag_pixel_y-5),(0,0,255),5,cv2.LINE_AA)
        cv2.putText(img, robot_tag_r, (tag_pixel_x1,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.putText(img, str(ROBOT_red[i]), (tag_pixel_x1+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
        #蓝方
        robot_tag_b='BS'
        
        cv2.putText(img, robot_tag_b, (tag_pixel_x2,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        cv2.line(img,(tag_pixel_x2+30,tag_pixel_y-5),(int(tag_pixel_x2+30+50*blood_percent_b),tag_pixel_y-5),(255,0,0),5,cv2.LINE_AA)
        cv2.putText(img, str(ROBOT_blue[i]), (tag_pixel_x2+50,tag_pixel_y), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)

'''line函数说明：第一个参数表示目标图片数据，
二表示线段起始位置,三表示终止位置，四表示线段的颜色'''
cv2.imshow('img',img)#展示绘制结果
cv2.waitKey(0)
