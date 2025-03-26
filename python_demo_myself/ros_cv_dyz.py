#coding=utf-8
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from ctypes import *
import sys
import threading
import time

sys.path.append("/opt/MVS/Samples/64/Python/MvImport")
from MvCameraControl_class import *

class CameraPublisher:
    def __init__(self):
        rospy.init_node('camera_publisher', anonymous=True)
        
        # 创建图像发布者和相机信息发布者
        self.image_pub = rospy.Publisher('image_topic', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
        
        # 相机参数配置
        self.camera_matrix = np.array([
            [5.45871341e+03, 0.00000000e+00, 1.48831894e+03],
            [0.00000000e+00, 5.43980951e+03, 8.68191166e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        
        self.dist_coeffs = np.array([
            -2.84309695e-02, 7.89779564e-01, -5.56054348e-03, 
            -4.42259917e-03, 1.96227336e+01
        ])
        
        # 默认参数
        self.gain_value = 19.0  # 默认增益值
        self.exposure_time = 55000.0  # 默认曝光时间(μs)
        
        # 初始化相机
        self.init_camera()

    def init_camera(self):
        # 枚举设备
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            rospy.logerr("enum devices fail! ret[0x%x]" % ret)
            return False
        if deviceList.nDeviceNum == 0:
            rospy.logerr("find no device!")
            return False

        # 选择设备并创建句柄
        self.cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            rospy.logerr("create handle fail! ret[0x%x]" % ret)
            return False

        # 打开设备
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            rospy.logerr("open device fail! ret[0x%x]" % ret)
            return False

        # 设置相机参数
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            rospy.logerr("set trigger mode fail! ret[0x%x]" % ret)
            return False

        # 设置初始曝光时间
        ret = self.set_exposure(self.exposure_time)
        if ret != 0:
            rospy.logwarn("set exposure fail! ret[0x%x]" % ret)

        # 设置初始增益值
        ret = self.set_gain(self.gain_value)
        if ret != 0:
            rospy.logwarn("set gain fail! ret[0x%x]" % ret)

        # 获取数据包大小
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            rospy.logerr("get payload size fail! ret[0x%x]" % ret)
            return False
        self.nPayloadSize = stParam.nCurValue

        # 开始取流
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            rospy.logerr("start grabbing fail! ret[0x%x]" % ret)
            return False

        return True

    def set_exposure(self, exposure_time):
        """设置相机曝光时间(单位:μs)"""
        # 获取曝光时间范围
        stFloatParam = MVCC_FLOATVALUE()
        memset(byref(stFloatParam), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam)
        if ret != 0:
            rospy.logerr("get exposure range fail! ret[0x%x]" % ret)
            return ret
        
        # 检查曝光时间是否在有效范围内
        min_exposure = stFloatParam.fMin
        max_exposure = stFloatParam.fMax
        if exposure_time < min_exposure or exposure_time > max_exposure:
            rospy.logwarn(f"Exposure time {exposure_time} out of range [{min_exposure}, {max_exposure}]")
            exposure_time = max(min_exposure, min(exposure_time, max_exposure))
        
        # 设置曝光时间
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret == 0:
            self.exposure_time = exposure_time
            rospy.loginfo(f"Set exposure time to {exposure_time}μs successfully")
        else:
            rospy.logerr(f"set exposure fail! ret[0x%x]" % ret)
        
        return ret

    def set_gain(self, gain_value):
        """设置相机增益"""
        # 获取增益范围
        stFloatParam = MVCC_FLOATVALUE()
        memset(byref(stFloatParam), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("Gain", stFloatParam)
        if ret != 0:
            rospy.logerr("get gain range fail! ret[0x%x]" % ret)
            return ret
        
        # 检查增益值是否在有效范围内
        min_gain = stFloatParam.fMin
        max_gain = stFloatParam.fMax
        if gain_value < min_gain or gain_value > max_gain:
            rospy.logwarn(f"Gain value {gain_value} out of range [{min_gain}, {max_gain}]")
            gain_value = max(min_gain, min(gain_value, max_gain))
        
        # 设置增益值
        ret = self.cam.MV_CC_SetFloatValue("Gain", gain_value)
        if ret == 0:
            self.gain_value = gain_value
            rospy.loginfo(f"Set gain to {gain_value} successfully")
        else:
            rospy.logerr(f"set gain fail! ret[0x%x]" % ret)
        
        return ret

    def create_camera_info_msg(self, frame):
        """创建相机参数消息"""
        camera_info = CameraInfo()
        camera_info.header = Header()
        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = "camera_frame"
        
        # 设置相机参数
        camera_info.height = frame.shape[0]
        camera_info.width = frame.shape[1]
        camera_info.distortion_model = "plumb_bob"
        
        # 相机内参矩阵 (3x3 row-major matrix)
        camera_info.K = self.camera_matrix.flatten().tolist()
        
        # 投影矩阵 (3x4 row-major matrix)
        camera_info.P = [
            self.camera_matrix[0,0], 0, self.camera_matrix[0,2], 0,
            0, self.camera_matrix[1,1], self.camera_matrix[1,2], 0,
            0, 0, 1, 0
        ]
        
        # 畸变参数 (k1, k2, t1, t2, k3)
        camera_info.D = self.dist_coeffs.flatten().tolist()
        
        # 通常R矩阵为单位矩阵
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        
        return camera_info

    def run(self):
        data_buf = (c_ubyte * self.nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            # 从相机取一帧图片
            ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, self.nPayloadSize, stFrameInfo, 1000)
            if ret == 0:
                # 将图像数据转换为numpy数组
                frame_data = np.frombuffer(data_buf, dtype=np.uint8)
                frame = frame_data.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))

                # 图像格式转换
                if stFrameInfo.enPixelType == 0x01080001:  # 灰度图像
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif stFrameInfo.enPixelType == 0x02180015:  # Bayer格式图像
                    frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 去畸变
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

                # 发布图像消息
                ros_image = self.cv2_to_imgmsg(frame)
                self.image_pub.publish(ros_image)

                # 发布相机参数
                camera_info = self.create_camera_info_msg(frame)
                self.camera_info_pub.publish(camera_info)
            else:
                rospy.logwarn("no data[0x%x]" % ret)

            rate.sleep()

    def cv2_to_imgmsg(self, cv_image):
        img_msg = Image()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.data = cv_image.tostring()
        img_msg.step = len(img_msg.data) // img_msg.height
        return img_msg

    def shutdown(self):
        # 停止取流
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            rospy.logerr("stop grabbing fail! ret[0x%x]" % ret)

        # 关闭设备
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            rospy.logerr("close device fail! ret[0x%x]" % ret)

        # 销毁句柄
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            rospy.logerr("destroy handle fail! ret[0x%x]" % ret)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        publisher = CameraPublisher()
        
        # 示例：动态调整曝光时间
        # publisher.set_exposure(10000.0)  # 设置为10ms
        
        publisher.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        publisher.shutdown()