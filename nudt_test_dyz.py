# coding=utf-8
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
        
        # 创建图像发布者和相机信息发布者（保持原样）
        self.image_pub = rospy.Publisher('image_topic', Image, queue_size=10)
        self.camera_info_pub = rospy.Publisher('camera_info', CameraInfo, queue_size=10)
        print("Image and camera info published")
        # 相机参数（保持原样）
        self.camera_matrix = np.array([
            [4.74121246e+03, 0.00000000e+00, 1.54304828e+03],
            [0.00000000e+00, 4.74257863e+03, 1.04017365e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.dist_coeffs = np.array([
            1.77935891e-02, 6.50135435e-02, -6.90728155e-04, 
            -4.89124420e-04, 2.79849057e+00
        ])


        self.gain_value = 5.0  # 默认增益值
        self.exposure_time = 55000.0  # 默认曝光时间(μs)
        # 初始化相机（保持原样）
        self.init_camera()

    def init_camera(self):
        # 枚举设备（保持原样）
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            rospy.logerr("enum devices fail! ret[0x%x]" % ret)
            return False
        if deviceList.nDeviceNum == 0:
            rospy.logerr("find no device!")
            return False

        # 选择设备并创建句柄（保持原样）
        self.cam = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            rospy.logerr("create handle fail! ret[0x%x]" % ret)
            return False

        # 打开设备（保持原样）
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            rospy.logerr("open device fail! ret[0x%x]" % ret)
            return False

        # ==================== 关键修改部分 ====================
        # 强制设置彩色输出格式（新增）
        ret = self.cam.MV_CC_SetEnumValue("PixelFormat", 0x02180014)  # BGR8格式
        if ret != 0:
            rospy.logwarn("set BGR8 format failed, trying Bayer...")
            ret = self.cam.MV_CC_SetEnumValue("PixelFormat", 0x02180015)  # 尝试Bayer格式
            if ret != 0:
                rospy.logerr("cannot set color format! Camera may be grayscale only")
        # ==================== 修改结束 ====================

        # 其他相机设置（保持原样）
        ret = self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            rospy.logerr("set trigger mode fail! ret[0x%x]" % ret)
            return False

        # 获取数据包大小（保持原样）
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            rospy.logerr("get payload size fail! ret[0x%x]" % ret)
            return False
        self.nPayloadSize = stParam.nCurValue

        # 开始取流（保持原样）
        # 在此之前设置曝光和增益
        self.set_exposure(self.exposure_time)
        self.set_gain(self.gain_value)

        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            rospy.logerr("start grabbing fail! ret[0x%x]" % ret)
            return False

        return True

    def run(self):
        data_buf = (c_ubyte * self.nPayloadSize)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

        rate = rospy.Rate(30)  # 30Hz

        while not rospy.is_shutdown():
            # 获取图像（保持原样）
            ret = self.cam.MV_CC_GetOneFrameTimeout(data_buf, self.nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                rospy.logwarn("no data[0x%x]" % ret)
                continue

            # ==================== 关键修改部分 ====================
            # 图像格式转换（优化彩色处理）
            frame = np.frombuffer(data_buf, dtype=np.uint8)
            frame = frame.reshape((stFrameInfo.nHeight, stFrameInfo.nWidth, -1))
            
            # 根据实际格式处理（新增调试输出）
            rospy.logdebug(f"PixelType: 0x{stFrameInfo.enPixelType:08x}")
            
            if stFrameInfo.enPixelType == 0x01080001:  # 灰度图像
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # 转伪彩色
                rospy.logwarn("Grayscale input detected! Check camera config")
            elif stFrameInfo.enPixelType == 0x02180015:  # Bayer格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BayerBG2BGR)  # 必须转换
            else:  # 其他情况默认按BGR处理
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 统一转RGB
            
            # 去畸变（保持原样）
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            # ==================== 修改结束 ====================

            # 发布图像（保持原样）
            ros_image = Image()
            ros_image.header = Header(stamp=rospy.Time.now(), frame_id="camera_frame")
            ros_image.height = frame.shape[0]
            ros_image.width = frame.shape[1]
            ros_image.encoding = "bgr8"  # 保持BGR编码
            ros_image.is_bigendian = 0
            ros_image.data = frame.tobytes()
            ros_image.step = len(ros_image.data) // ros_image.height
            
            self.image_pub.publish(ros_image)
            
            # 发布相机信息（保持原样）
            camera_info = self.create_camera_info_msg(frame)
            self.camera_info_pub.publish(camera_info)
            
            rate.sleep()

    # 以下方法完全保持原样
    def create_camera_info_msg(self, frame):
        camera_info = CameraInfo()
        camera_info.header = Header()
        camera_info.header.stamp = rospy.Time.now()
        camera_info.header.frame_id = "camera_frame"
        camera_info.height = frame.shape[0]
        camera_info.width = frame.shape[1]
        camera_info.distortion_model = "plumb_bob"
        camera_info.K = self.camera_matrix.flatten().tolist()
        camera_info.P = [
            self.camera_matrix[0,0], 0, self.camera_matrix[0,2], 0,
            0, self.camera_matrix[1,1], self.camera_matrix[1,2], 0,
            0, 0, 1, 0
        ]
        camera_info.D = self.dist_coeffs.flatten().tolist()
        camera_info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        return camera_info

    def set_exposure(self, exposure_time):
        stFloatParam = MVCC_FLOATVALUE()
        memset(byref(stFloatParam), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", stFloatParam)
        if ret != 0:
            rospy.logerr("get exposure range fail! ret[0x%x]" % ret)
            return ret
        min_exposure = stFloatParam.fMin
        max_exposure = stFloatParam.fMax
        if exposure_time < min_exposure or exposure_time > max_exposure:
            exposure_time = max(min_exposure, min(exposure_time, max_exposure))
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        if ret == 0:
            self.exposure_time = exposure_time
            rospy.loginfo(f"Set exposure time to {exposure_time}μs successfully")
        else:
            rospy.logerr(f"set exposure fail! ret[0x%x]" % ret)
        return ret

    def set_gain(self, gain_value):
        stFloatParam = MVCC_FLOATVALUE()
        memset(byref(stFloatParam), 0, sizeof(MVCC_FLOATVALUE))
        ret = self.cam.MV_CC_GetFloatValue("Gain", stFloatParam)
        if ret != 0:
            rospy.logerr("get gain range fail! ret[0x%x]" % ret)
            return ret
        min_gain = stFloatParam.fMin
        max_gain = stFloatParam.fMax
        if gain_value < min_gain or gain_value > max_gain:
            gain_value = max(min_gain, min(gain_value, max_gain))
        ret = self.cam.MV_CC_SetFloatValue("Gain", gain_value)
        if ret == 0:
            self.gain_value = gain_value
            rospy.loginfo(f"Set gain to {gain_value} successfully")
        else:
            rospy.logerr(f"set gain fail! ret[0x%x]" % ret)
        return ret

    def shutdown(self):
        ret = self.cam.MV_CC_StopGrabbing()
        if ret != 0:
            rospy.logerr("stop grabbing fail! ret[0x%x]" % ret)
        ret = self.cam.MV_CC_CloseDevice()
        if ret != 0:
            rospy.logerr("close device fail! ret[0x%x]" % ret)
        ret = self.cam.MV_CC_DestroyHandle()
        if ret != 0:
            rospy.logerr("destroy handle fail! ret[0x%x]" % ret)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        publisher = CameraPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        publisher.shutdown()