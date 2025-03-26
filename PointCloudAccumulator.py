from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import time

import rospy


class PointCloudAccumulator:
    def __init__(self, accumulation_time, publish_topic):
        self.accumulation_time = accumulation_time
        self.publish_topic = publish_topic
        self.points = []
        self.last_received_time = time.time()

        # ros node初始化
        rospy.init_node("point_cloud_accumulator", anonymous=True)

        # 订阅激光雷达点云数据
        rospy.Subscriber("livox/lidar", PointCloud2, self.callback)

        # 发布积累后的稠密点云
        self.pub_dense_point_cloud = rospy.Publisher(self.publish_topic, PointCloud2, queue_size=10)

    def publish_dense_point_cloud(self):
        # 创建新的PointCloud2消息
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "your_frame_id"
        dense_point_cloud_msg = pc2.create_cloud_xyz32(header, self.points)

        #发布稠密点云
        self.pub_dense_point_cloud.publish(dense_point_cloud_msg)


    def callback(self,point_cloud_msg):
        # 获取点云时间戳
        current_time = time.time()

        # 将PointCloud2数据转换为numpy数组
        pc_data = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(pc_data))

        # 将新的点云数据添加到积累列表中
        self.points.extend(points)

        # 如果积累时间超过设定的阈值， 发布稠密点云
        if (current_time-self.last_received_time) > self.accumulation_time:
            self.publish_dense_point_cloud()
            print("已发布点云，话题为：",self.publish_topic)
            self.points = []
            self.last_received_time = current_time

if __name__ == "__main__":
    accumulation_time = 1 # 积累时间， 单位： 秒
    publish_topic = "/dense_point_cloud_topic"
    accumulator = PointCloudAccumulator(accumulation_time, publish_topic)
    # 循环监听ROS消息
    rospy.spin()












