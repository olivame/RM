import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import open3d as o3d

from pylab import *

import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import matplotlib.pylab as plt
class yolov5_detector:
    def __init__(self, weight_path, img_size=(640, 640), conf_thres=0.70, iou_thres=0.2, max_det=10,
                 device='', classes=None, agnostic_nms=False, augment=False, visualize=False, half=False, dnn=False,data='data/coco128.yaml'):
        self.device = select_device(device)
        self.weight_path_first = weight_path
        self.model = DetectMultiBackend(self.weight_path_first, device=self.device, dnn=dnn, fp16=False)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.img_size = check_img_size(img_size, s=self.stride)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.augment = augment
        self.visualize = visualize
        self.half = half  # 需要修改
        bs = 1
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.img_size))  # warmup
    def predict(self,img):
        im0 = img.copy()
        im = letterbox(im0, self.img_size, self.model.stride, auto=self.model.pt)[0]  # np.array()
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # 半精度推理需要修改
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        # NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        detections = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = self.names[int(cls)]
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()
                    xywh = [round(x) for x in xywh]
                    xywh = [xywh[0] - xywh[2] // 2, xywh[1] - xywh[3] // 2, xywh[2], xywh[3]]
                    confidence = float(conf)
                    detections.append((label, xywh, confidence))
        return detections
class nudt_lidar_exp:
    def __init__(self):
        R_tc = np.float64([
            [-0.21013,  -0.03920,  0.97689],
            [-0.97767,   0.01045, -0.20988],
            [-0.00198,  -0.99918, -0.04052]
        ])
        t_tc = np.float64([[-0.10632], [-0.08475], [-0.01299]])

        # 求逆得到点云到相机
        R_ct = R_tc.T
        t_ct = -R_tc.T @ t_tc
        self.Rotate_matrix = R_ct
        self.tvec = t_ct.flatten()
        self.rvec, _ = cv2.Rodrigues(self.Rotate_matrix)
        # 相机内参和畸变参数保持原样
        self.camera_matrix = np.float64([[4.74121246e+03, 0.00000000e+00, 1.54304828e+03],
                                         [0.00000000e+00, 4.74257863e+03, 1.04017365e+03],
                                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        self.distCoeffs = np.float64([1.77935891e-02, 6.50135435e-02, -6.90728155e-04, -4.89124420e-04, 2.79849057e+00])
        print("初始化完毕,等待点云发布\n")
    '''     self.Rotate_matrix = np.float64([[-0.195049, -0.980605, -0.0192303],
                                         [0.0239639, 0.0148363, -0.999603],
                                         [0.980501, -0.195432, 0.0206054]])
        self.rvec, _ = cv2.Rodrigues(self.Rotate_matrix)
        # 经过排序修改后得到的平移矩阵
        self.tvec = np.float64([-0.16477, -0.0466339, 0.052998])
        # 相机内部参数,matlab
        self.camera_matrix = np.float64([[1947, 0, 1576.9],
                                         [0, 1946.9, 1061.2],
                                         [0, 0, 1]])
        # 相机形变矩阵
        self.distCoeffs = np.float64([-0.4509, 0.2993, -0.00009985, 0.0001312, -0.1297])'''
        
    def pointcloud_callback(self,msg):
        print("接收到点云，开始处理\n")
        t1 = time.time()
        pointcloud = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        point_cloud_list = [point for point in pointcloud]
        point_cloud_np = o3d.utility.Vector3dVector(point_cloud_list)
        pc_o3d = o3d.geometry.PointCloud(point_cloud_np)
        self.cloud_ndarray = np.array(pc_o3d.points)
        t2 = time.time()
        print(f"点云处理完毕,处理耗时：{t2 - t1}\n")
    def img_callback(self, msg):
        img_np = np.frombuffer(msg.data, dtype=np.uint8)
        frame = img_np.reshape((msg.height, msg.width, 3)).copy()  # 关键：添加 .copy() 确保可写
        self.frame = frame
    
nudt_lidar = nudt_lidar_exp()
rospy.init_node("nudt_lidar_v4", anonymous=True)
msg = rospy.wait_for_message("/dense_point_cloud_topic", PointCloud2)
nudt_lidar.pointcloud_callback(msg)
t3 = time.time()
point_2d, _ = cv2.projectPoints(nudt_lidar.cloud_ndarray, nudt_lidar.rvec, nudt_lidar.tvec, nudt_lidar.camera_matrix, nudt_lidar.distCoeffs)
t4 = time.time()
print(f"点云投影耗时：{t4 - t3}\n")

msg = rospy.wait_for_message("/image_topic", Image)
nudt_lidar.img_callback(msg)
h_frame, w_frame = nudt_lidar.frame.shape[0:2]
print(f"图像的尺寸为：{(h_frame, w_frame)}\n")
x_2d_list = []
y_2d_list = []
distance_list = []
index = -1
for point in point_2d:
    index += 1
    if (0 <= point[0][0] <= w_frame) and (0 <= point[0][1] <= h_frame) and (0 <= index <= len(nudt_lidar.cloud_ndarray)):
        distance = np.sqrt(nudt_lidar.cloud_ndarray[index, 0]**2 +
                           nudt_lidar.cloud_ndarray[index, 1]**2 +
                           nudt_lidar.cloud_ndarray[index, 2]**2)
        x_2d_list.append(point[0][0])
        y_2d_list.append(point[0][1])
        distance_list.append(distance)
try:
    color_pp = max(distance_list) - min(distance_list)
except:
    color_pp = 1000000
    print("error_defined_1")
for i in range(len(x_2d_list)):
    try:
        color = int(((distance_list[i])*255)/color_pp)
    except:
        color = 0
    cv2.circle(nudt_lidar.frame, (int(x_2d_list[i]), int(y_2d_list[i])), 1, (0, 0, color), 3)

cv2.namedWindow("frame", 0)
cv2.imshow("frame", nudt_lidar.frame)
# 修改此处，添加按下"q"退出功能
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyWindow("frame")

# 加载模型
weights_path = "/home/olivame/yolov5_camera_final_1/weights_test/r1/r1_728_s.pt"
detector = yolov5_detector(weights_path, img_size=(3088, 2064), data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14)
weights_path_second = "/home/olivame/yolov5_camera_final_1/weights_test/r2/r2_727_s.pt"

detector_second = yolov5_detector(weights_path_second, data='armor/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=1)
def img_pro_callback(msg):
    t1 = time.time()
    image_np = np.frombuffer(msg.data, dtype=np.uint8)
    frame = image_np.reshape((msg.height, msg.width, 3))
    results = detector.predict(frame)
    annotator = Annotator(np.ascontiguousarray(frame), line_width=3, example=str(detector_second.names))
    if len(results) > 0:
        for detection in results:
            label, xywh, confidence = detection
            x, y, w, h = xywh
            x, y, w, h = int(x), int(y), int(w), int(h)
            img2 = frame[y:y + h, x:x + w]
            results_second = detector_second.predict(img2)
            if len(results_second) > 0:
                for detection_second in results_second:
                    label_second, xywh_second, confidence_second = detection_second
                    xyxy_second_in_img1 = []
                    x1_2, y1_2, w_2, h_2 = xywh_second
                    x1 = torch.tensor(x + x1_2, device=detector.device)
                    y1 = torch.tensor(y + y1_2, device=detector.device)
                    x2 = torch.tensor(x + x1_2 + w_2, device=detector.device)
                    y2 = torch.tensor(y + y1_2 + h_2, device=detector.device)
                    xyxy_second_in_img1.append(x1)
                    xyxy_second_in_img1.append(y1)
                    xyxy_second_in_img1.append(x2)
                    xyxy_second_in_img1.append(y2)
                # 计算点云平均位置
                iou_count = 0
                m = -1
                mean_cloud = np.float64([0, 0, 0])  # yolo所识别到物体点云平均位置
                for point in point_2d:
                    m = m + 1
                    if (x <= point[0][0] <= x + w) and (y <= point[0][1] <= y + h):
                        mean_cloud += nudt_lidar.cloud_ndarray[m]
                        iou_count += 1
                mean_cloud = mean_cloud / iou_count
                print(f"点云平均位置为：{mean_cloud}")
                annotator.box_label(xyxy_second_in_img1, label_second, color=colors(int(confidence_second), True))
    t2 = time.time()
    print(f"处理一帧耗时为：{t2 - t1}")
    img = annotator.result()
    cv2.namedWindow("Detect", 0)
    cv2.imshow("Detect", img)
    # 修改此处，添加按下"q"退出功能
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyWindow("Detect")
        rospy.signal_shutdown("User pressed q to exit.")

rospy.Subscriber("/image_topic", Image, img_pro_callback)
rospy.spin()
"""
点云计算效率低
"""