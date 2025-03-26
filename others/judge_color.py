"""
created by wangjiming
"""
import cv2
import random
import numpy as np


'''#注释部分为通过识别灯条判段红蓝（弃用）
def is_valid_lightblob(contours, rect):      #判断是否为有效灯条
    width, height = rect[1]
    if width*height < 10:
        return False
    lw_rate = width/height if width > height else height/width  #矩形长宽比
    areaRatio = cv2.contourArea(contours) / (width*height)
    ret = (lw_rate>1.2 and lw_rate<10) and areaRatio>0.7
    return ret
    



def get_lightblobs(armor_bin):  #获取灯条轮廓
    lightblobs = []
    contours, hierarchy = cv2.findContours(armor_bin,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if(hierarchy[0][i][2] == -1):
            rect = cv2.minAreaRect(contours[i])
            if(is_valid_lightblob(contours[i],rect)):
                lightblobs.append(contours[i])
    return lightblobs


def get_lightblobs_area(lightblobs_counter): #获取灯条轮廓的总面积
    area = 0
    for l_c in lightblobs_counter:
        area += cv2.contourArea(l_c)
    return area
			


def red_or_blue(armor_src):  #通过灯条识别灯条来判断是红方还是蓝方
    armor_bin_red = img_preprocess(armor_src, 'red')
    lightblobs_red = get_lightblobs(armor_bin_red)  #返回是否存在灯条数
    n_lightblobs_red = len(lightblobs_red)
    #print('lights_red: %d'%n_lightblobs_red)
    #cv2.imshow('armor_bin_red',armor_bin_red)
    armor_bin_blue = img_preprocess(armor_src, 'blue')
    lightblobs_blue = get_lightblobs(armor_bin_blue)
    n_lightblobs_blue = len(lightblobs_blue)
    #print('lights_blue: %d'%n_lightblobs_blue)
    #cv2.imshow('armor_bin_blue',armor_bin_blue)
    lightblobs_area_red = get_lightblobs_area(lightblobs_red)
    lightblobs_area_blue = get_lightblobs_area(lightblobs_blue)
    
    if(n_lightblobs_red > n_lightblobs_blue or lightblobs_area_red > lightblobs_area_blue): #红色灯条数，轮廓面积大于蓝色灯条数判定为红色
        return 1
    elif(n_lightblobs_red < n_lightblobs_blue or lightblobs_area_red < lightblobs_area_blue):
        return -1
    else:
        return 1


def diff_self_enemy(xyxys, clses, names, src, enemy):
    proimage0 = src.copy()
    self_enemy = np.zeros(len(xyxys))   #用来存储每个矩形框是红方还是己方（大于0表示红，小于0表示蓝方）通过权重加分来判定
    for i in range(len(clses)):
        cls = names[clses[i]]
        xyxy = xyxys[i]
        if(cls == 'armor' or cls == 'ignore'):
            armor_src = proimage0[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]]
            self_enemy[i] += red_or_blue(armor_src)
            #cv2.imshow('armor_src',armor_src)
            #cv2.waitKey(0)
            for j in range(len(clses)):
                cls_t = names[clses[j]]
                if (cls_t != 'armor' and cls_t != 'ignore'):
                    xyxy_t = xyxys[j]
                    c = ((xyxy[2]+xyxy[0])/2, (xyxy[3]+xyxy[1])/2)   #装甲板的中心坐标
                    if(c[0]< xyxy_t[2] and c[0] > xyxy_t[0] and c[1] < xyxy_t[3] and c[1] > xyxy_t[1]):   #装甲板中心在车的矩形框里
                        self_enemy[j] += self_enemy[i]
                        break

    for i in range(len(clses)):
        if (self_enemy[i]==0): #还不确定的矩形框，再进行一次灯条判断
            xyxy = xyxys[i]
            roi = proimage0[xyxy[1]:xyxy[3],xyxy[0]:xyxy[2]]
            self_enemy[i] += red_or_blue(roi)
            
    for i in range(len(clses)):
        cls_i = names[clses[i]]
        if (cls_i == 'armor' or cls_i == 'ignore'):
            xyxy_i = xyxys[i]
            for j in range(i,len(clses)):
                cls_j = names[clses[j]]
                if(cls_j != 'armor' or cls_j != 'ignore'):
                    xyxy_j = xyxys[j]
                    c = ((xyxy_i[2]+xyxy_i[0])/2, (xyxy_i[3]+xyxy_i[1])/2)   #装甲板的中心坐标
                    if(c[0]< xyxy_j[2] and c[0] > xyxy_j[0] and c[1] < xyxy_j[3] and c[1] > xyxy_j[1]):
                        self_enemy[i] = 1 if self_enemy[j] > 0 else -1

    if (enemy=='red'):
        return self_enemy > 0
    else:
        return self_enrmy < 0
'''


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




def img_preprocess(armor_src, c):

    ###############################wjm#########################################
    b, g, r = cv2.split(armor_src)
    if(c == 'red'):
        armor_gray = r
    elif (c == 'blue'):
        armor_gray = b
    else:
        armor_gray = g
    armor_gray = cv2.medianBlur(armor_gray, 3)   #中值滤波
    ret, armor_bin = cv2.threshold(armor_gray, 225, 255, cv2.THRESH_BINARY) #二值化阈值

    # ##############################softword###########################################
    #
    # hsv = cv2.cvtColor(armor_src, cv2.COLOR_BGR2HSV)
    # if (c == 'red'):
    #     lower_red1 = np.array([0, 43, 46])
    #     higher_red1 = np.array([10, 255, 255])
    #     lower_red2 = np.array([156, 43, 46])
    #     higher_red2 = np.array([180, 255, 255])
    #     mask1 = cv2.inRange(hsv, lower_red1, higher_red1)
    #     mask2 = cv2.inRange(hsv, lower_red2, higher_red2)
    #     armor_bin = cv2.bitwise_or(mask1, mask2)
    # else:
    #     lower_blue = np.array([100, 43, 46])
    #     higher_blue = np.array([124, 255, 255])
    #     armor_bin = cv2.inRange(hsv, lower_blue, higher_blue)
    # #############################################################################


    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))#定义结构元素的形状和大小
    # armor_bin = cv2.dilate(armor_bin, dilate_kernel)#膨胀操作
    #
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))#定义结构元素的形状和大小
    # armor_bin = cv2.erode(armor_bin, erode_kernel)#腐蚀操作
    #
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))#定义结构元素的形状和大小
    # armor_bin = cv2.dilate(armor_bin, dilate_kernel)#膨胀操作
    #
    # erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))#定义结构元素的形状和大小
    # armor_bin = cv2.erode(armor_bin, erode_kernel)#腐蚀操作
    
    return armor_bin


def red_or_blue(img_bin_red, img_bin_blue):   #红色返回0，蓝色返回1
    area_red = get_all_counter_area(img_bin_red)
    #print(area_red)
    area_bule = get_all_counter_area(img_bin_blue)
    #print(area_bule)
    if area_red > area_bule:
        return 0
    else:
        return 1
    
def get_all_counter_area(armor_bin):  #获取所有轮廓的面积
    area = 0
    contours, hierarchy = cv2.findContours(armor_bin,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        # x,y,w,h=cv2.boundingRect(contours[i])
        # if(hierarchy[0][i][2] == -1 and w/h<2): #w/h排除哨兵无敌状态上方白色灯条的影响
        #     area += cv2.contourArea(contours[i])
        if(hierarchy[0][i][2] == -1 ): #w/h排除哨兵无敌状态上方白色灯条的影响
            area += cv2.contourArea(contours[i])
    return area


def get_colors(xyxys, src):     #红色返回0，蓝色返回1
    img_temp = src.copy()
    colors = []
    for i in range(len(xyxys)):
        xyxy = xyxys[i]
        #截取矩形框图片
        img_xyxy = img_temp[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])]
        #图像预处理
        img_bin_red = img_preprocess(img_xyxy,'red')
        img_bin_blue = img_preprocess(img_xyxy,'blue')
        #查看预处理效果
        img_bin_red = cv2.resize(img_bin_red, (600, 400))
        img_bin_blue = cv2.resize(img_bin_blue, (600, 400))

        # cv2.namedWindow("red")  # 创建一个image的窗口
        # cv2.imshow('red',img_bin_red)
        # cv2.namedWindow("blue")  # 创建一个image的窗口
        # cv2.imshow('blue',img_bin_blue)
        #判断颜色（二值化处理后，通过比较总的轮廓面积来判断颜色，简单）
        colors.append(red_or_blue(img_bin_red, img_bin_blue))
        #cv2.imshow('img',img_xyxy)
        #cv2.imshow('red',img_bin_red)
        #cv2.imshow('blue',img_bin_blue)
        #cv2.waitKey(0)
        
    return colors
	
	
if __name__ == '__main__':
    names = ['car', 'watcher', 'base', 'ignore', 'armor']
    img_path = 'test/test_red.png'
    input_file = 'test/2.txt'
    img = cv2.imread(img_path)
    img_scr = img.copy()
    info_file = open(input_file)  
    lines = info_file.readlines()
    enemy = 'red'
    bbox_xyxys = []
    clses = []
    for line in lines:
        line = line.split()
        cls = int(line[0])
        # xyxy = (int(line[1]),int(line[2]),int(line[3]),int(line[4]))
        xyxy = (int(line[1]), int(line[2]), int(line[3]), int(line[4]))
        clses.append(cls)
        bbox_xyxys.append(xyxy)

    results = get_colors(bbox_xyxys,img_scr)
    print(results)
    colors = [(255,0,0) if i else (0,0,255) for i in results]
    for i in range(len(bbox_xyxys)):
        x = bbox_xyxys[i]
        cls = names[clses[i]]
        #if(cls == 'armor'):
        plot_one_box(bbox_xyxys[i],img_scr, color = colors[i],line_thickness = 2)
    cv2.imwrite('test/red_wjm.jpg', img_scr)
    cv2.imshow('img_scr',img_scr)
    
    cv2.waitKey(0)
