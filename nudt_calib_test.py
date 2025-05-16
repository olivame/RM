#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
import open3d as o3d
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2
import time

class CalibrationVerifier:
    def __init__(self):
        # 初始化外参参数（替换为您的实际参数）
        R_tc = np.float64([
            [-0.21013,  -0.03920,  0.97689],
            [-0.97767,   0.01045, -0.20988],
            [-0.00198,  -0.99918, -0.04052]
        ])
        t_tc = np.float64([[-0.10632], [-0.08475], [-0.01299]])

        # 求逆得到点云到相机
        R_ct = R_tc.T
        t_ct = -R_tc.T @ t_tc
        self.R = R_ct
        self.t = t_ct.flatten()
        self.rvec, _ = cv2.Rodrigues(self.R)
        
        # 相机内参（替换为您的实际参数）
        self.camera_matrix = np.array([
            [4741.21246, 0, 1543.04828],
            [0, 4742.57863, 1040.17365],
            [0, 0, 1]
        ])
        self.dist_coeffs = np.array([1.15934654e-02  ,3.52621432e-01 ,-4.29806809e-04 ,-2.65678230e-04,
  -5.74771859e-01])
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
        # 棋盘格参数
        self.chessboard_size = (8, 11)  # 内部角点数 (width, height)
        self.square_size = 0.03  # 方格边长（单位：米）

        # 数据存储
        self.current_cloud = None
        self.current_image = None

        # ROS订阅
        rospy.Subscriber("/dense_point_cloud_topic", PointCloud2, self.cloud_callback)
        rospy.Subscriber("/image_topic", Image, self.image_callback)

    def cloud_callback(self, msg):
        """ 处理点云数据 """
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.current_cloud = np.array(list(points))
        print("点云坐标示例（前5个点）:", self.current_cloud[20000:20005])
    def image_callback(self, msg):
        """ 处理图像数据 """
        self.current_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

    def verify_calibration(self):
        """ 执行标定验证 """
        while not rospy.is_shutdown():
            if self.current_cloud is None or self.current_image is None:
                rospy.loginfo("等待数据...")
                time.sleep(1)
                continue

            # 1. 检测图像中的棋盘格
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            ret, img_corners = cv2.findChessboardCorners(gray, self.chessboard_size, 
                                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                              cv2.CALIB_CB_NORMALIZE_IMAGE)
            
            if not ret:
                rospy.logwarn("未检测到棋盘格！请调整位置")
                time.sleep(1)
                continue

            # 2. 生成3D棋盘格角点（世界坐标系，假设Z=0）
            obj_points = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), dtype=np.float32)
            obj_points[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1,2) * self.square_size

            # 3. 从点云中提取棋盘格平面
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.current_cloud)
            
            # 使用RANSAC拟合平面
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
            plane_points = np.asarray(pcd.points)[inliers]

            if len(plane_points) < 100:
                rospy.logwarn("棋盘格平面点太少！请检查点云质量")
                continue

            # 4. 将3D角点对齐到拟合的平面
            # 计算平面到世界坐标系的变换
            plane_normal = plane_model[:3]
            rotation = self._rotation_matrix_from_vectors([0,0,1], plane_normal)
            plane_center = np.mean(plane_points, axis=0)
            
            # 变换角点到点云平面（世界坐标系下的点云角点）
            chessboard_corners_3d = (rotation @ obj_points.T).T + plane_center

            # =========================
            # 若你想“把图片上的角点投影到点云平面”：
            # 1. 需要知道每个像素角点的深度（Z），否则无法唯一反投影到3D。
            # 2. 若假设棋盘格平面方程已知，可用相机内参和外参将像素点反投影到该平面。
            # 下面是一个简单实现（假设所有角点都在拟合平面上）：
            # =========================
            # img_corners: (N,1,2) -> (N,2)
            img_corners_2d = img_corners.squeeze(1)
            plane_normal = plane_model[:3]
            d = -np.dot(plane_normal, plane_center)
            # 相机坐标系到世界坐标系的旋转和平移
            R = self.R
            t = self.t
            # 相机内参
            K = self.camera_matrix

            img_corners_3d_on_plane = []
            for pt in img_corners_2d:
                # 像素坐标 -> 相机归一化坐标
                uv1 = np.array([pt[0], pt[1], 1.0])
                ray_cam = np.linalg.inv(K) @ uv1
                # 相机坐标系下的射线
                ray_cam = ray_cam / np.linalg.norm(ray_cam)
                # 世界坐标系下的射线
                ray_world = R.T @ ray_cam
                cam_center_world = -R.T @ t
                # 射线和平面求交
                denom = np.dot(plane_normal, ray_world)
                if abs(denom) < 1e-6:
                    img_corners_3d_on_plane.append([np.nan, np.nan, np.nan])
                    continue
                t_intersect = -(np.dot(plane_normal, cam_center_world) + d) / denom
                p_world = cam_center_world + t_intersect * ray_world
                img_corners_3d_on_plane.append(p_world)
            img_corners_3d_on_plane = np.array(img_corners_3d_on_plane)
            # img_corners_3d_on_plane 即为图片角点反投影到点云平面的三维坐标
            # =========================

            # 5. 将点云平面上的角点（世界坐标系下）通过外参和内参投影到图像
            # 这是“点云到图片”的投影，和你原本的逻辑一致
            projected_points, _ = cv2.projectPoints(
                chessboard_corners_3d,  # 世界坐标系下的点云角点
                self.rvec, self.t,      # 点云到相机的外参（R_ct, t_ct）
                self.camera_matrix, self.dist_coeffs  # 相机内参
            )

            # 6. 计算误差
            error = np.mean(np.linalg.norm(img_corners - projected_points, axis=2))
            rospy.loginfo(f"标定误差: {error:.2f} pixels")

            # 7. 可视化
            vis_img = self.current_image.copy()
            cv2.drawChessboardCorners(vis_img, self.chessboard_size, img_corners, ret)
            # 绘制投影点（绿色）和检测点（红色）
            for i, (img_pt, proj_pt) in enumerate(zip(img_corners, projected_points)):
                img_pt = tuple(img_pt[0].astype(int))
                proj_pt = tuple(proj_pt[0].astype(int))
                cv2.circle(vis_img, img_pt, 5, (0, 0, 255), -1)  # 红色：检测到的角点
                cv2.circle(vis_img, proj_pt, 5, (0, 255, 0), 2)   # 绿色：投影的角点
                cv2.line(vis_img, img_pt, proj_pt, (255, 0, 0), 1)  # 蓝色连线

            cv2.imshow("Calibration Verification", vis_img)
            cv2.waitKey(100)

    def _rotation_matrix_from_vectors(self, vec1, vec2):
        """ 计算两个向量间的旋转矩阵 """
        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        
        if s == 0:
            return np.eye(3)
        
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

if __name__ == "__main__":
    rospy.init_node("calibration_verifier")
    verifier = CalibrationVerifier()
    
    # 等待第一帧数据
    rospy.loginfo("等待初始数据...")
    while verifier.current_cloud is None or verifier.current_image is None:
        time.sleep(0.1)
    
    rospy.loginfo("开始标定验证")
    verifier.verify_calibration()