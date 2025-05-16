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
        # 初始化外参参数
        self.R = np.array([[-0.205674, 0.00965221, 0.978573],
                          [-0.978592, 0.00560896, -0.205733],
                          [-0.00747455, -0.999938, 0.00829195]])
        self.t = np.array([0.2, 0.0312579, 0.0833853])  # 初始平移向量
        self.rvec, _ = cv2.Rodrigues(self.R)
        
        # 相机内参
        self.camera_matrix = np.array([[2314.77843, 0, 1540.88249],
                                      [0, 5310.85510, 1018.01056],
                                      [0, 0, 1]])
        self.dist_coeffs = np.array([1.15934654e-02, 3.52621432e-01, -4.29806809e-04, 
                                    -2.65678230e-04, -5.74771859e-01])

        # 棋盘格参数
        self.chessboard_size = (8, 11)
        self.square_size = 0.03

        # 数据存储
        self.current_cloud = None
        self.current_image = None
        self.obj_points = None  # 存储3D角点

        # 优化参数
        self.learning_rate = 0.001  # 学习率
        self.max_iterations = 10000   # 最大迭代次数
        self.tolerance = 1e-4       # 收敛阈值

        # ROS订阅
        rospy.Subscriber("/dense_point_cloud_topic", PointCloud2, self.cloud_callback)
        rospy.Subscriber("/image_topic", Image, self.image_callback)

    def cloud_callback(self, msg):
        """ 处理点云数据 """
        points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.current_cloud = np.array(list(points))

    def image_callback(self, msg):
        """ 处理图像数据 """
        self.current_image = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))

    def compute_reprojection_error(self, t):
        """ 计算给定平移向量时的重投影误差 """
        _, img_corners = self.find_chessboard()
        if img_corners is None:
            return float('inf')
        
        projected_points, _ = cv2.projectPoints(self.transformed_obj_points, 
                                              self.rvec, t,
                                              self.camera_matrix, 
                                              self.dist_coeffs)
        return np.mean(np.linalg.norm(img_corners - projected_points, axis=2))

    def optimize_translation(self):
        """ 使用梯度下降优化平移向量 """
        best_t = self.t.copy()
        best_error = float('inf')
        
        for iteration in range(self.max_iterations):
            # 计算当前误差
            current_error = self.compute_reprojection_error(best_t)
            
            # 计算数值梯度
            gradient = np.zeros(3)
            delta = 0.001  # 微分量
            
            for i in range(3):
                t_plus = best_t.copy()
                t_plus[i] += delta
                error_plus = self.compute_reprojection_error(t_plus)
                
                t_minus = best_t.copy()
                t_minus[i] -= delta
                error_minus = self.compute_reprojection_error(t_minus)
                
                gradient[i] = (error_plus - error_minus) / (2 * delta)
            
            # 更新平移向量
            new_t = best_t - self.learning_rate * gradient
            new_error = self.compute_reprojection_error(new_t)
            
            # 检查是否收敛
            if abs(new_error - current_error) < self.tolerance:
                break
                
            if new_error < best_error:
                best_t = new_t
                best_error = new_error
            else:
                # 如果误差没有改善，减小学习率
                self.learning_rate *= 0.5
        
        return best_t, best_error

    def find_chessboard(self):
        """ 检测棋盘格并返回角点 """
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        return cv2.findChessboardCorners(gray, self.chessboard_size, 
                                        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                              cv2.CALIB_CB_NORMALIZE_IMAGE)

    def verify_calibration(self):
        """ 执行标定验证 """
        while not rospy.is_shutdown():
            if self.current_cloud is None or self.current_image is None:
                rospy.loginfo("等待数据...")
                time.sleep(1)
                continue

            # 1. 检测图像中的棋盘格
            ret, img_corners = self.find_chessboard()
            if not ret:
                rospy.logwarn("未检测到棋盘格！请调整位置")
                time.sleep(1)
                continue

            # 2. 生成3D棋盘格角点
            self.obj_points = np.zeros((self.chessboard_size[0]*self.chessboard_size[1], 3), 
                                      dtype=np.float32)
            self.obj_points[:,:2] = np.mgrid[0:self.chessboard_size[0], 
                                           0:self.chessboard_size[1]].T.reshape(-1,2) * self.square_size

            # 3. 从点云中提取棋盘格平面
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.current_cloud)
            
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.02, 
                                                    ransac_n=3, 
                                                    num_iterations=1000)
            plane_points = np.asarray(pcd.points)[inliers]

            if len(plane_points) < 100:
                rospy.logwarn("棋盘格平面点太少！请检查点云质量")
                continue

            # 4. 将3D角点对齐到拟合的平面
            plane_normal = plane_model[:3]
            rotation = self._rotation_matrix_from_vectors([0,0,1], plane_normal)
            plane_center = np.mean(plane_points, axis=0)
            
            self.transformed_obj_points = (rotation @ self.obj_points.T).T + plane_center

            # 5. 优化平移向量
            optimized_t, optimized_error = self.optimize_translation()
            
            # 更新平移向量
            self.t = optimized_t
            rospy.loginfo(f"优化后的平移向量: {self.t}, 误差: {optimized_error:.2f} pixels")

            # 6. 使用优化后的参数进行投影
            projected_points, _ = cv2.projectPoints(self.transformed_obj_points, 
                                                  self.rvec, self.t, 
                                                  self.camera_matrix, 
                                                  self.dist_coeffs)

            # 7. 可视化
            vis_img = self.current_image.copy()
            cv2.drawChessboardCorners(vis_img, self.chessboard_size, img_corners, ret)
            vis_img = cv2.resize(vis_img, (800, 600))
            
            for i, (img_pt, proj_pt) in enumerate(zip(img_corners, projected_points)):
                img_pt = tuple(img_pt[0].astype(int))
                proj_pt = tuple(proj_pt[0].astype(int))
                cv2.circle(vis_img, img_pt, 5, (0, 0, 255), -1)
                cv2.circle(vis_img, proj_pt, 5, (0, 255, 0), 2)
                cv2.line(vis_img, img_pt, proj_pt, (255, 0, 0), 1)

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
    
    rospy.loginfo("等待初始数据...")
    while verifier.current_cloud is None or verifier.current_image is None:
        time.sleep(0.1)
    
    rospy.loginfo("开始标定验证")
    verifier.verify_calibration()