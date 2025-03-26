#coding=utf-8
import cv2
import numpy as np
import mvsdk
import platform
import rospy
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge, CvBridgeError

def main_loop(image_pub):
	# 枚举相机
	DevList = mvsdk.CameraEnumerateDevice()
	nDev = len(DevList)
	if nDev < 1:
		print("No camera was found!")
		return

	for i, DevInfo in enumerate(DevList):
		print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
	i = 0 if nDev == 1 else int(input("Select camera: "))
	DevInfo = DevList[i]
	print(DevInfo)

	# 打开相机
	hCamera = 0
	try:
		hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
	except mvsdk.CameraException as e:
		print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
		return

	# 获取相机特性描述
	cap = mvsdk.CameraGetCapability(hCamera)

	# 判断是黑白相机还是彩色相机
	monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

	# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
	if monoCamera:
		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
	else:
		mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

	# 相机模式切换成连续采集
	mvsdk.CameraSetTriggerMode(hCamera, 0)

	# 手动曝光，曝光时间30ms -> 调整亮度
	mvsdk.CameraSetAeState(hCamera, 0)
	mvsdk.CameraSetExposureTime(hCamera, 60 * 1000)
	mvsdk.CameraSetGamma(hCamera,70)
	mvsdk.CameraSetContrast(hCamera,105)
	mvsdk.CameraSetAnalogGain(hCamera,100)
	mvsdk.CameraSetGain(hCamera,100,100,100)
	mvsdk.CameraSetSaturation(hCamera,100)

	# 让SDK内部取图线程开始工作
	mvsdk.CameraPlay(hCamera)

	# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
	FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

	# 分配RGB buffer，用来存放ISP输出的图像
	# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
	pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
	P = [[1947, 0, 1576.9],
		 [0, 1946.9, 1061.2],
		 [0, 0, 1]]
	K = [-0.4509, 0.2993, -0.00009985, 0.0001312, -0.1297]
	while (cv2.waitKey(1) & 0xFF) != ord('q'):
		# 从相机取一帧图片
		try:
			pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
			mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
			mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

			# windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
			# linux下直接输出正的，不需要上下翻转
			if platform.system() == "Windows":
				mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)
			
			# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
			# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
			frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
			frame = np.frombuffer(frame_data, dtype=np.uint8)
			frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
			frame = cv2.undistort(frame, np.array(P), np.array(K))

			# frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
			ros_image = cv2_to_imgmsg(frame)
			# 发布图像消息到话题上
			# rate = rospy.Rate(1) # 1Hz
			image_pub.publish(ros_image)
			# rate.sleep()
			
			# cv2.imshow("Press q to end", frame)
			
		except mvsdk.CameraException as e:
			if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
				print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message) )

	# 关闭相机
	mvsdk.CameraUnInit(hCamera)

	# 释放帧缓存
	mvsdk.CameraAlignFree(pFrameBuffer)

"""
    Provides conversions between OpenCV and ROS image formats in a hard-coded way.  
    CV_Bridge, the module usually responsible for doing this, is not compatible with Python 3,
     - the language this all is written in.  So we create this module, and all is... well, all is not well,
     - but all works.  :-/
"""
import sys
import numpy as np
from sensor_msgs.msg import Image

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
		# bridge = CvBridge()
		main_loop(image_pub)
	finally:
		cv2.destroyAllWindows()

main()
