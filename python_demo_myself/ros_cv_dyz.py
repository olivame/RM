#coding=utf-8
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from ctypes import *
import sys
import threading

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")  # 导入相应SDK的库，实际安装位置绝对路径
from MvCameraControl_class import *

def main_loop(image_pub):
    # 枚举设备
    deviceList = MV_CC_DEVICE_INFO_LIST()
    tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
    ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
    if ret != 0:
        print("enum devices fail! ret[0x%x]" % ret)
        return
    if deviceList.nDeviceNum == 0:
        print("find no device!")
        return
    print("Find %d devices!" % deviceList.nDeviceNum)

    # 选择设备并创建句柄
    cam = MvCamera()
    stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(stDeviceList)
    if ret != 0:
        print("create handle fail! ret[0x%x]" % ret)
        return

    # 打开设备
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        print("open device fail! ret[0x%x]" % ret)
        return

    # 设置相机参数
    ret = cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    if ret != 0:
        print("set trigger mode fail! ret[0x%x]" % ret)
        return

    # 获取数据包大小
    stParam = MVCC_INTVALUE()
    memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
    ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
    if ret != 0:
        print("get payload size fail! ret[0x%x]" % ret)
        return
    nPayloadSize = stParam.nCurValue

    # 开始取流
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        print("start grabbing fail! ret[0x%x]" % ret)
        return

    # 分配缓冲区
    data_buf = (c_ubyte * nPayloadSize)()

    # 设置标定参数
    '''P = [[1947, 0, 1576.9],
         [0, 1946.9, 1061.2],
         [0, 0, 1]]
    K = [-0.4509, 0.2993, -0.00009985, 0.0001312, -0.1297]'''
    #Camera Matrix (K):
    P = [[5.45871341e+03,0.00000000e+00,1.48831894e+03],
    [0.00000000e+00,5.43980951e+03,8.68191166e+02],
    [0.00000000e+00,0.00000000e+00,1.00000000e+00]]
    #Distortion Coefficients:
    K = [[-2.84309695e-02 ,7.89779564e-01,-5.56054348e-03,-4.42259917e-03,1.96227336e+01]]
    stFrameInfo = MV_FRAME_OUT_INFO_EX()
    memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        # 从相机取一帧图片
        ret = cam.MV_CC_GetOneFrameTimeout(data_buf, nPayloadSize, stFrameInfo, 1000)
        if ret == 0:
            # 将图像数据转换为numpy数组
            frame_data = np.frombuffer(data_buf, dtype=np.uint8)
            frame = frame_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))

            # 如果是灰度图像，转换为BGR
            if stFrameInfo.enPixelType == 0x01080001:  # 灰度图像
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif stFrameInfo.enPixelType == 0x02180015:  # Bayer格式图像
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 去畸变
            frame = cv2.undistort(frame, np.array(P), np.array(K))

            # 发布图像消息到话题上
            ros_image = cv2_to_imgmsg(frame)
            image_pub.publish(ros_image)

        else:
            print("no data[0x%x]" % ret)

    # 停止取流
    ret = cam.MV_CC_StopGrabbing()
    if ret != 0:
        print("stop grabbing fail! ret[0x%x]" % ret)

    # 关闭设备
    ret = cam.MV_CC_CloseDevice()
    if ret != 0:
        print("close device fail! ret[0x%x]" % ret)

    # 销毁句柄
    ret = cam.MV_CC_DestroyHandle()
    if ret != 0:
        print("destroy handle fail! ret[0x%x]" % ret)

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr("This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8") # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3), # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                    dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def cv2_to_imgmsg(cv_image):
    img_msg = Image()
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.data = cv_image.tostring()
    img_msg.step = len(img_msg.data) // img_msg.height # That double line is actually integer division, not a comment
    return img_msg

def main():
    try:
        rospy.init_node('image_publisher', anonymous=True)
        image_pub = rospy.Publisher('image_topic', Image, queue_size=10)
        main_loop(image_pub)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()