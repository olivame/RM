import rospy
import time
import numpy as np
import cv2
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, Image

def load_calibration():
    """
    加载已知的旋转矩阵和平移向量，以及相机内参和畸变参数
    """
    # lidar->camera 外参（已知）
    R_tc = np.array([
        [-0.21013,  -0.03920,  0.97689],
        [-0.97767,   0.01045, -0.20988],
        [-0.00198,  -0.99918, -0.04052]
    ], dtype=np.float64)
    t_tc = np.array([[-0.10632], [-0.08475], [-0.01299]], dtype=np.float64)
    # 求逆得到 pc -> cam 外参
    R_ct = R_tc.T
    t_ct = -R_ct @ t_tc
    # 相机内参
    K = np.array([[4.74121246e+03, 0, 1.54304828e+03],
                  [0, 4.74257863e+03, 1.04017365e+03],
                  [0, 0, 1]], dtype=np.float64)
    # 畸变系数
    dist = np.array([1.77935891e-02, 6.50135435e-02,
                     -6.90728155e-04, -4.89124420e-04, 2.79849057e+00],
                    dtype=np.float64)
    return R_ct, t_ct.flatten(), K, dist

def project_and_show(pc_msg, img_msg):
    # 1. 点云转 numpy (N,3)
    pc_iter = point_cloud2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=True)
    pc = np.array(list(pc_iter), dtype=np.float64)
    # 2. 图像转 numpy
    img = np.frombuffer(img_msg.data, np.uint8).reshape(img_msg.height, img_msg.width, 3)
    # 3. 加载标定参数
    R, t, K, dist = load_calibration()
    # 4. 投影
    pts2d, _ = cv2.projectPoints(pc, cv2.Rodrigues(R)[0], t, K, dist)
    pts2d = pts2d.reshape(-1,2)
    # 5. 按距离给点上色（生成连续色图）
    depths = np.linalg.norm(pc, axis=1)
    dmin, dmax = depths.min(), depths.max()
    norms = ((depths - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)

    # 使用 COLORMAP_INFERNO：近处 → 深色，远处 → 浅色
    cmap = cv2.COLORMAP_INFERNO
    # applyColorMap 要求输入为 2D 单通道图，输出为 (N,1,3)
    colors = cv2.applyColorMap(norms.reshape(-1,1), cmap)[:,0,:]

    # 6. 在图像上画点
    overlay = img.copy()
    h, w = img.shape[:2]
    for (u, v), color in zip(pts2d, colors):
        ui, vi = int(u + 0.5), int(v + 0.5)
        if 0 <= ui < w and 0 <= vi < h:
            b, g, r = int(color[0]), int(color[1]), int(color[2])
            cv2.circle(overlay, (ui, vi), 2, (b, g, r), -1)
    # 7. 显示结果
    window = 'Calibration Validation'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, overlay)
    print("按下 q 键退出...")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('calibration_validator', anonymous=True)
    print("等待一帧点云和一帧图像...")
    pc_msg = rospy.wait_for_message('/dense_point_cloud_topic', PointCloud2)
    img_msg = rospy.wait_for_message('/image_topic', Image)
    t0 = time.time()
    project_and_show(pc_msg, img_msg)
    print(f"可视化完成，用时 {time.time() - t0:.3f}s")