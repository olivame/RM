"""
1. 整理前期代码
"""
import numpy as np
from pylab import *

import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
import time
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
import open3d as o3d

####################################通信部分起始###########################################################
import struct
import mmap
import serial
CRC8_TAB = [0x00, 0x5e, 0xbc, 0xe2, 0x61, 0x3f, 0xdd, 0x83, 0xc2, 0x9c, 0x7e, 0x20, 0xa3, 0xfd, 0x1f, 0x41,
            0x9d, 0xc3, 0x21, 0x7f, 0xfc, 0xa2, 0x40, 0x1e, 0x5f, 0x01, 0xe3, 0xbd, 0x3e, 0x60, 0x82, 0xdc,
            0x23, 0x7d, 0x9f, 0xc1, 0x42, 0x1c, 0xfe, 0xa0, 0xe1, 0xbf, 0x5d, 0x03, 0x80, 0xde, 0x3c, 0x62,
            0xbe, 0xe0, 0x02, 0x5c, 0xdf, 0x81, 0x63, 0x3d, 0x7c, 0x22, 0xc0, 0x9e, 0x1d, 0x43, 0xa1, 0xff,
            0x46, 0x18, 0xfa, 0xa4, 0x27, 0x79, 0x9b, 0xc5, 0x84, 0xda, 0x38, 0x66, 0xe5, 0xbb, 0x59, 0x07,
            0xdb, 0x85, 0x67, 0x39, 0xba, 0xe4, 0x06, 0x58, 0x19, 0x47, 0xa5, 0xfb, 0x78, 0x26, 0xc4, 0x9a,
            0x65, 0x3b, 0xd9, 0x87, 0x04, 0x5a, 0xb8, 0xe6, 0xa7, 0xf9, 0x1b, 0x45, 0xc6, 0x98, 0x7a, 0x24,
            0xf8, 0xa6, 0x44, 0x1a, 0x99, 0xc7, 0x25, 0x7b, 0x3a, 0x64, 0x86, 0xd8, 0x5b, 0x05, 0xe7, 0xb9,
            0x8c, 0xd2, 0x30, 0x6e, 0xed, 0xb3, 0x51, 0x0f, 0x4e, 0x10, 0xf2, 0xac, 0x2f, 0x71, 0x93, 0xcd,
            0x11, 0x4f, 0xad, 0xf3, 0x70, 0x2e, 0xcc, 0x92, 0xd3, 0x8d, 0x6f, 0x31, 0xb2, 0xec, 0x0e, 0x50,
            0xaf, 0xf1, 0x13, 0x4d, 0xce, 0x90, 0x72, 0x2c, 0x6d, 0x33, 0xd1, 0x8f, 0x0c, 0x52, 0xb0, 0xee,
            0x32, 0x6c, 0x8e, 0xd0, 0x53, 0x0d, 0xef, 0xb1, 0xf0, 0xae, 0x4c, 0x12, 0x91, 0xcf, 0x2d, 0x73,
            0xca, 0x94, 0x76, 0x28, 0xab, 0xf5, 0x17, 0x49, 0x08, 0x56, 0xb4, 0xea, 0x69, 0x37, 0xd5, 0x8b,
            0x57, 0x09, 0xeb, 0xb5, 0x36, 0x68, 0x8a, 0xd4, 0x95, 0xcb, 0x29, 0x77, 0xf4, 0xaa, 0x48, 0x16,
            0xe9, 0xb7, 0x55, 0x0b, 0x88, 0xd6, 0x34, 0x6a, 0x2b, 0x75, 0x97, 0xc9, 0x4a, 0x14, 0xf6, 0xa8,
            0x74, 0x2a, 0xc8, 0x96, 0x15, 0x4b, 0xa9, 0xf7, 0xb6, 0xe8, 0x0a, 0x54, 0xd7, 0x89, 0x6b, 0x35]
wCRC_Table = [0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf, 0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c,
              0xdbe5, 0xe97e, 0xf8f7,
              0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e, 0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed,
              0xcb64, 0xf9ff, 0xe876,
              0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd, 0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e,
              0xfae7, 0xc87c, 0xd9f5,
              0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c, 0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef,
              0xea66, 0xd8fd, 0xc974,
              0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb, 0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868,
              0x99e1, 0xab7a, 0xbaf3,
              0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a, 0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9,
              0x8960, 0xbbfb, 0xaa72,
              0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9, 0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a,
              0xb8e3, 0x8a78, 0x9bf1,
              0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738, 0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb,
              0xa862, 0x9af9, 0x8b70,
              0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7, 0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64,
              0x5fed, 0x6d76, 0x7cff,
              0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036, 0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5,
              0x4f6c, 0x7df7, 0x6c7e,
              0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5, 0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66,
              0x7eef, 0x4c74, 0x5dfd,
              0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134, 0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7,
              0x6e6e, 0x5cf5, 0x4d7c,
              0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3, 0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60,
              0x1de9, 0x2f72, 0x3efb,
              0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232, 0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1,
              0x0d68, 0x3ff3, 0x2e7a,
              0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1, 0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62,
              0x3ceb, 0x0e70, 0x1ff9,
              0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330, 0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3,
              0x2c6a, 0x1ef1, 0x0f78]


class Port:
    def __init__(self, port, band):
        self.port = port
        self.baud = int(band)
        self.__open_port = None
        self.get_data_flag = True
        self.__real_time_all_data = b''  # 接受的所有数据
        self.__real_time_one_data = b''  # 一次接受的数据

    def get_real_time_all_data(self):
        ret = self.__real_time_all_data
        self.__real_time_all_data = b''
        return ret

    def get_real_time_one_data(self):
        return self.__real_time_one_data

    def clear_real_time_data(self):
        self.__real_time_all_data = b''

    # 设置是否接收数据
    def set_get_data_flag(self, get_data_flag):
        self.get_data_flag = get_data_flag

    def openPort(self):
        try:
            self.__open_port = serial.Serial(self.port, self.baud)
            threading.Thread(target=self.get_data, args=()).start()  # 开启数据接收线程
            print("Open port successfully")
        except Exception as e:
            print('Open com fail:{}/{}'.format(self.port, self.baud))
            print('Exception:{}'.format(e))

    def close(self):
        if self.__open_port is not None and self.__open_port.isOpen:
            self.get_data_flag = False
            self.__open_port.close()

    def send_data(self, data):
        if self.__open_port is None:
            self.openPort()
        if self.__open_port is None:
            print("Warring: Port is not None!")
            return 0
        if self.__open_port.isOpen():
            success_bytes = self.__open_port.write(data)
            return success_bytes
        else:
            print("Warring: Port is closed!")

    def get_data(self):
        if self.__open_port is None:
            self.openPort()
        while self.__open_port.isOpen():
            if self.get_data_flag:
                n = self.__open_port.inWaiting()
                if n:
                    data = self.__open_port.read()  # 一次读所有的字节
                    # print('当前向串口传输的数据为：',data)         #显示
                    self.__real_time_all_data += data
                    self.__real_time_one_data = data
        # print('向串口传输的数据为：',self.__real_time_all_data)
        print("Warning: Port is closed!")


uint8_t = 'B'
uint16_t = 'H'
uint32_t = 'I'


# 接口协议说明请参考《裁判系统学生串口协议附录》,在python中，数据流以bytes类型处理
class Communicator:
    def __init__(self):
        """

        Returns
        -------
        object
        """
        self.header_SOF = b'\xa5'
        self.CRC8_INIT = b'\xff'
        self.CRC16_INIT = b'\xff\xff'
        self.header_length = 5
        self.storage_mode = 'little'  # 数据存储的大小端模式
        self.Alig_format = '>'
        self.isCRC = True  # 数据接收时是否校验
        self.ROBOT_BLOODS = []
        self.cmd_id_blood = int(0x0003).to_bytes(2, 'little')
        self.cmd_id_command = int(0x0303).to_bytes(2, 'little')
        self.cmd_id_status = int(0x0001).to_bytes(2, 'little')
        self.cmd_id_position = int(0x0203).to_bytes(2, 'little')
        self.cmd_id_mark = int(0x020C).to_bytes(2, 'little')

    def set_isCRC(self, BOOL):
        self.is_CRC = BOOL

    def command_send(self, sender_ID, receiver_ID, command_x_f, command_y_f, command_z_f, command_key, command_ID,
                     seq):  # 云台手信息,seq为0
        cmd_id = int(0x0301).to_bytes(2, self.storage_mode)
        data_flag = int(0x0201).to_bytes(2, self.storage_mode)
        sender_ID_data = int(sender_ID).to_bytes(2, self.storage_mode)
        receiver_ID_data = int(receiver_ID).to_bytes(2, self.storage_mode)
        pos_x = struct.pack('f', command_x_f)
        pos_y = struct.pack('f', command_y_f)
        pos_z = struct.pack('f', command_z_f)
        key_send = int(command_key).to_bytes(1, self.storage_mode)
        ID_send = int(command_ID).to_bytes(1, self.storage_mode)
        data = data_flag + sender_ID_data + receiver_ID_data + pos_x + pos_y + pos_z + key_send + ID_send

        # ID_send=int(command_ID).to_bytes(1,self.storage_mode)
        # ID = int(robot_ID).to_bytes(2,self.storage_mode)
        # # x = int(robot_x).to_bytes(4,self.storage_mode,signed=True)
        # # y = int(robot_y).to_bytes(4,self.storage_mode,signed=True)
        # x=struct.pack('f',robot_x)
        # y=struct.pack('f',robot_y)
        # toward = struct.pack('f',robot_toward)
        # data = ID + x + y + toward
        # cmd_id = int(0x0305).to_bytes(2,self.storage_mode)
        return self.package_data(cmd_id, seq, data)

    def lmap_interaction_send(self, pos_data, seq):
        cmd_id = int(0x0305).to_bytes(2, self.storage_mode)
        data = b''  # 初始化一个空的字节序列
        # pos_data中存储所有的坐标
        # 将每个数据从pos_data中取出并打包为2字节的字节序列
        for i in range(0, len(pos_data), 2):
            x = pos_data[i].to_bytes(2, self.storage_mode)
            y = pos_data[i + 1].to_bytes(2, self.storage_mode)
            data += x + y  # 将x和y的字节序列连接到data中

        return self.package_data(cmd_id, seq, data)

        # def lmap_interaction_send(self, robot_ID, robot_x, robot_y, robot_toward, seq):  # 小地图交互
        #	ID = int(robot_ID).to_bytes(2, self.storage_mode)
        #	# x = int(robot_x).to_bytes(4,self.storage_mode,signed=True)
        #	# y = int(robot_y).to_bytes(4,self.storage_mode,signed=True)
        #	x = struct.pack('f', robot_x)
        #	y = struct.pack('f', robot_y)
        #	toward = struct.pack('f', robot_toward)
        #	data = ID + x + y + toward
        #	cmd_id = int(0x0305).to_bytes(2, self.storage_mode)
        #	return self.package_data(cmd_id, seq, data)
        """
    operate_tpye  0:空操作 
                  1:删除图层
                  2: 删除所有
    layer：       0~9
        """

    # 发送盲区预测点坐标
    def send_point_guess(robot_ID, guess_time_limit):
        # print(guess_value_now.get(robot_ID),guess_value.get(robot_ID) ,guess_index[robot_ID])
        # 进度未满 and 预测进度没有涨 and 超过单点预测时间上限，同时满足则切换另一个点预测

        return 0

    def read_time(self, data):
        time_left = int.from_bytes(data, 'little')
        return time_left

    def recieve_position(self):
        shmem = mmap.mmap(0, 100, 'global_share_memory_position', mmap.ACCESS_READ)
        s = str(shmem.read(shmem.size()).decode("utf-8"))
        return s

    # vs2012早期版本会有截断符和开始符号，需要提取有用字符串
    # es='\\x00'#字符条件截断，还没有设计开始endstring
    # if s.find(es)==-1:
    #     print(s)
    # else:
    #     sn=s[:s.index(s)]
    #     print('other data')

    # 读取标记信息
    def read_mark(self, com):
        s = com.get_real_time_all_data()
        s = self.unpackage_data(s)
        # print("read blood:")
        for data_pack in s:
            # cmd_id = int.from_bytes(data_pack[5:7], com_class.storage_mode)
            cmd_id = data_pack[5:7]
            # print('cmd_id为',cmd_id)
            # cmd_id=data_pack[5:7].to_bytes(2,self.storage_mode)
            if cmd_id == self.cmd_id_mark:
                # print("标记进度")
                mark1 = int.from_bytes(data_pack[7:8], 'little')  # 对方英雄
                mark2 = int.from_bytes(data_pack[8:9], 'little')  # 对方工程
                mark3 = int.from_bytes(data_pack[9:10], 'little')  # 对方3号步兵
                mark4 = int.from_bytes(data_pack[11:12], 'little')  # 对方4号步兵
                mark5 = int.from_bytes(data_pack[12:13], 'little')  # 对方5号步兵
                mark6 = int.from_bytes(data_pack[13:14], 'little')  # 对方哨兵
                mark_all = [mark1, mark2, mark3, mark4, mark5, mark6]
                return mark_all

    def read_blood(self, data):
        blood_R1 = int.from_bytes(data[0:2], 'little')
        self.ROBOT_BLOODS.append(blood_R1)  # 0
        # print('红1英雄机器人血量为',blood_R1)
        blood_R2 = int.from_bytes(data[2:4], 'little')
        self.ROBOT_BLOODS.append(blood_R2)  # 1
        # print('红2工程机器人血量为',blood_R2)
        blood_R3 = int.from_bytes(data[4:6], 'little')
        self.ROBOT_BLOODS.append(blood_R3)  # 2
        # print('红3步兵机器人血量为',blood_R3)
        blood_R4 = int.from_bytes(data[6:8], 'little')
        self.ROBOT_BLOODS.append(blood_R4)  # 3
        # print('红4步兵机器人血量为',blood_R4)
        blood_R5 = int.from_bytes(data[8:10], 'little')
        self.ROBOT_BLOODS.append(blood_R5)  # 4
        # print('红5步兵机器人血量为',blood_R5)
        blood_R7 = int.from_bytes(data[10:12], 'little')
        self.ROBOT_BLOODS.append(blood_R7)  # 5
        # print('红7哨兵机器人血量为',blood_R7)
        blood_R_POST = int.from_bytes(data[12:14], 'little')
        self.ROBOT_BLOODS.append(blood_R_POST)  # 6
        # print('红方前哨站血量为',blood_R_POST)
        blood_R_BASE = int.from_bytes(data[14:16], 'little')
        self.ROBOT_BLOODS.append(blood_R_BASE)  # 7
        # print('红方基地血量为',blood_R_BASE)
        blood_B1 = int.from_bytes(data[16:18], 'little')
        self.ROBOT_BLOODS.append(blood_B1)  # 8
        # print('蓝1英雄机器人血量为',blood_B1)
        blood_B2 = int.from_bytes(data[18:20], 'little')
        self.ROBOT_BLOODS.append(blood_B2)  # 9
        # print('蓝2工程机器人血量为',blood_B2)
        blood_B3 = int.from_bytes(data[20:22], 'little')
        self.ROBOT_BLOODS.append(blood_B3)  # 10
        # print('蓝3步兵机器人血量为',blood_B3)
        blood_B4 = int.from_bytes(data[22:24], 'little')
        self.ROBOT_BLOODS.append(blood_B4)  # 11
        # print('蓝4步兵机器人血量为',blood_B4)
        blood_B5 = int.from_bytes(data[24:26], 'little')
        self.ROBOT_BLOODS.append(blood_B5)  # 12
        # print('蓝5步兵机器人血量为',blood_B5)
        blood_B7 = int.from_bytes(data[26:28], 'little')
        self.ROBOT_BLOODS.append(blood_B7)  # 13
        # print('蓝7哨兵机器人血量为',blood_B7)
        blood_B_POST = int.from_bytes(data[28:30], 'little')
        self.ROBOT_BLOODS.append(blood_B_POST)  # 14
        # print('蓝方前哨站血量为',blood_B_POST)
        blood_B_BASE = int.from_bytes(data[30:32], 'little')
        self.ROBOT_BLOODS.append(blood_B_BASE)  # 15
        return 0

    # print('蓝方基地血量为',blood_B_BASE)
    # send_blooddata=str(ROBOT_BLOODS)
    # #向C++通过共享内存传送血量信息
    # with contextlib.closing(mmap.mmap(-1, 1024, tagname='global_share_memory', access=mmap.ACCESS_WRITE)) as m:
    #     m.seek(0)
    #     m.write((send_blooddata).encode())
    #     m.flush()
    #     print (datetime.now(), "msg " + send_blooddata)
    #
    #     time.sleep(1)

    def analysis_data_pack(self, data_packs):  # 接收数据包分析  (有待完善)
        status_seconds = 0
        for data_pack in data_packs:
            # cmd_id = int.from_bytes(data_pack[5:7], self.storage_mode)
            cmd_id = data_pack[5:7]
            # print('cmd_id为',cmd_id)
            # cmd_id=data_pack[5:7].to_bytes(2,self.storage_mode)
            if cmd_id == self.cmd_id_blood:
                print("血量信息")
                blood_data = data_pack[7:35]
                self.read_blood(blood_data)
            if cmd_id == self.cmd_id_status:
                status_time = data_pack[8:9]
                status_seconds = int.from_bytes(status_time, 'little')
                if status_seconds != 0:
                    return status_seconds
            else:
                continue
        return status_seconds

    def package_data(self, cmd_id, seq, data):  # 封装数据包
        header_data_lenth = len(data).to_bytes(2, self.storage_mode)
        header_seq = int(seq).to_bytes(1, self.storage_mode)
        frame_header = self.header_SOF + header_data_lenth + header_seq + self.CRC8_INIT
        frame_header = self.append_CRC8_check_sum(frame_header, len(frame_header))
        if isinstance(cmd_id, int):
            cmd_id = cmd_id.to_bytes(2, self.storage_mode)
        if (frame_header is not None):
            package = frame_header + cmd_id + data + self.CRC16_INIT
            package = self.append_CRC16_check_sum(package, len(package))
            if (package is not None):
                return package
        return None

    def unpackage_data(self, data_bytes):  # 将数据流拆解为数据包
        data_packs = []
        n = len(data_bytes)
        for i in range(n):
            byte = data_bytes[i].to_bytes(1, self.storage_mode)
            if byte == self.header_SOF:
                # print('帧头为',byte)
                # m=data_bytes[i+1].to_bytes(1,self.storage_mode)
                # print(m)
                if i + 5 < n:
                    # cmd_id=data_bytes[i+5].to_bytes(2,self.storage_mode)
                    header = data_bytes[i:i + 5]
                    # cmd_id = data_bytes[i+5:i+7]
                    # print('第一次读',cmd_id)
                    # if cmd_id == self.cmd_id_blood:
                    #     print("血量信息")
                    # print(header)
                    data_length = int.from_bytes(header[1:3], byteorder=self.storage_mode)  # 将int类型转换为bytes类型，小端格式
                    # print(data_length)
                    # data=data_bytes[i+7:i+7+data_length]
                    if i + 5 + 2 + data_length + 2 <= n and (
                            (not self.isCRC) or (self.isCRC and self.verify_CRC8_check_sum(header, len(header)))):
                        data_pack = data_bytes[i:i + 5 + 2 + data_length + 2]
                        if (not self.isCRC) or (
                                self.isCRC and self.verify_CRC16_check_sum(data_pack, (len(data_pack)))):
                            data_packs.append(data_pack)
                            i += 5 + 2 + data_length + 2 - 1
        return data_packs

    def delete_graphic_frame(self, sender_ID, receiver_ID, operate_tpye, layer, seq):
        cmd_id = int(0x0301).to_bytes(2, self.storage_mode)
        data_cmd_id = int(0x0100).to_bytes(2, self.storage_mode)
        sender_ID = int(sender_ID).to_bytes(2, self.storage_mode)
        receiver_ID = int(receiver_ID).to_bytes(2, self.storage_mode)
        graphic_delete = struct.pack(self.Alig_format + uint8_t + uint8_t, \
                                     operate_tpye, layer)
        data = data_cmd_id + sender_ID + receiver_ID + graphic_delete
        return self.package_data(cmd_id, seq, data)

        """

        """

    def drawing_graphic_frame(self, sender_ID, receiver_ID, \
                              graphic_name, operate_tpye, graphic_tpye, layer, color, start_angle, end_angle, \
                              width, start_x, start_y, \
                              radius, end_x, end_y, \
                              seq):

        cmd_id = int(0x0301).to_bytes(2, self.storage_mode)
        data_cmd_id = int(0x0101).to_bytes(2, self.storage_mode)
        sender_ID = int(sender_ID).to_bytes(2, self.storage_mode)
        receiver_ID = int(receiver_ID).to_bytes(2, self.storage_mode)
        bit_mark1 = 32
        bit_mark2 = 32
        bit_mark3 = 32
        # 图形配置1
        bit_mark1 -= 3
        operate_tpye = int(operate_tpye) << bit_mark1
        bit_mark1 -= 3
        graphic_tpye = int(graphic_tpye) << bit_mark1
        bit_mark1 -= 4
        layer = int(layer) << bit_mark1
        bit_mark1 -= 4
        color = int(color) << bit_mark1
        bit_mark1 -= 9
        start_angle = int(start_angle) << bit_mark1
        bit_mark1 -= 9
        end_angle = int(end_angle) << bit_mark1
        graphic_config1 = operate_tpye | graphic_tpye | layer | color | start_angle | end_angle
        # 图形配置2
        bit_mark2 -= 10
        width = int(width) << bit_mark2
        bit_mark2 -= 11
        start_x = int(start_x) << bit_mark2
        bit_mark2 -= 11
        start_y = int(start_y) << bit_mark2
        graphic_config2 = width | start_x | start_y
        # 图形配置3
        bit_mark3 -= 10
        radius = int(radius) << bit_mark3
        bit_mark3 -= 11
        end_x = int(end_x) << bit_mark3
        bit_mark3 -= 11
        end_y = int(end_y) << bit_mark3
        graphic_config3 = radius | end_x | end_y
        # graphic_data
        graphic_data = struct.pack(self.Alig_format + '3s' + uint32_t + uint32_t + uint32_t, \
                                   graphic_name, graphic_config1, graphic_config2, graphic_config3)

        data = data_cmd_id + sender_ID + receiver_ID + graphic_data
        return self.package_data(cmd_id, seq, data)

    # CRC校验
    # crc8 generator polynomial:G(x)=x8+x5+x4+1
    """
    输入： bytes类型的pchMessage，int类型dwLength，bytes类型ucCRC8
    输出：bytes类型ucCRC8
    """

    def get_CRC8_check_sum(self, pchMessage, dwLength, ucCRC8):
        ucIndex = 0
        i = 0
        ucCRC8 = int.from_bytes(ucCRC8, self.storage_mode)
        while i < dwLength:
            ucIndex = ucCRC8 ^ (pchMessage[i])
            ucCRC8 = CRC8_TAB[ucIndex]
            i += 1
        return int(ucCRC8).to_bytes(1, self.storage_mode)

    def verify_CRC8_check_sum(self, pchMessage, dwLength):

        if int.from_bytes(pchMessage, self.storage_mode) == 0 or dwLength <= 2:
            return False
        ucExpected = self.get_CRC8_check_sum(pchMessage, dwLength - 1, self.CRC8_INIT)
        return ucExpected == pchMessage[dwLength - 1:]

    def append_CRC8_check_sum(self, pchMessage, dwLength):
        if int.from_bytes(pchMessage, self.storage_mode) == 0 or dwLength <= 2:
            return None
        ucCRC = self.get_CRC8_check_sum(pchMessage, dwLength - 1, self.CRC8_INIT)
        pchMessage = pchMessage[0:dwLength - 1] + ucCRC
        return pchMessage

    """
    输入： bytes类型的pchMessage，int类型dwLength，bytes类型wCRC
    输出：bytes类型wCRC
    """

    def get_CRC16_check_sum(self, pchMessage, dwLength, wCRC):
        chData = 0x00
        i = 0
        wCRC = int.from_bytes(wCRC, self.storage_mode)
        if (pchMessage == None):
            return int(0xFFFF).to_bytes(2, self.storage_mode)
        while i < dwLength:
            chData = pchMessage[i]
            wCRC = (wCRC >> 8) ^ wCRC_Table[(wCRC ^ chData) & 0x00ff]
            i += 1
        return int(wCRC).to_bytes(2, self.storage_mode)

    def verify_CRC16_check_sum(self, pchMessage, dwLength):
        wExpected = 0
        if pchMessage == None or dwLength <= 2:
            return False
        wExpected = self.get_CRC16_check_sum(pchMessage, dwLength - 2, self.CRC16_INIT)
        wExpected = int.from_bytes(wExpected, self.storage_mode)
        m2 = pchMessage[dwLength - 2]
        m1 = pchMessage[dwLength - 1]
        return ((wExpected & 0xff) == m2 and ((wExpected >> 8) & 0xff) == m1)

    def append_CRC16_check_sum(self, pchMessage, dwLength):
        if (pchMessage == None) or (dwLength <= 2):
            return None
        wCRC = self.get_CRC16_check_sum(pchMessage, dwLength - 2, self.CRC16_INIT)

        pchMessage = pchMessage[0:dwLength - 2] + wCRC
        return pchMessage
####################################通信部分结束###########################################################


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

class lidar_nudt:
    def __init__(self):
        # 右相机
        self.Rotate_matrix = np.float64([[-0.195049, -0.980605, -0.0192303],
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
        self.distCoeffs = np.float64([-0.4509, 0.2993, -0.00009985, 0.0001312, -0.1297])
        self.weights_path = "/home/ldk/RM_exp/code/yolov5_camera_final_1/weights_test/r1/best.pt"
        self.detector = yolov5_detector(self.weights_path,img_size= (3088, 2064), data='yaml/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=14)
        self.weights_path_second = "/home/ldk/RM_exp/code/yolov5_camera_final_1/weights_test/r2/r2_722.pt"
        self.detector_second = yolov5_detector(self.weights_path_second, data='armor/car.yaml', conf_thres=0.1, iou_thres=0.5, max_det=1)
        # 比赛设置
        self.T1 = True  # 用于判断mean_cloud是否有效
        self.enemy_is_blue = False
        self.lidar_pos_x = 1.628
        self.lidar_pos_y = 9.41
        # python中0为假
        self.guess_7_state = 0
        self.guess_7_decision = 0
        # 敌方总位置信息
        # 敌方为蓝方
        if self.enemy_is_blue:
            self.pos_data = [17.2, 4.6, 18.4, 4.2, 19.5, 13.5, 19.5, 13.5, 19.5, 13.5, 22.5, 7.5]
        else:
            # 敌方是红方
            self.pos_data = [11, 13.7, 9.6, 10.8, 8.5, 1.5, 8.5, 1.5, 8.5, 1.5, 5.7, 7.5]
        # 定时器一直发送
        # self.timer = rospy.Timer(rospy.Duration(0.2), self.timer_callback)
        print("初始化完毕")
    def pointcloud_callback(self, msg):
        # 将点云话题转换成open3d可以处理的类型
        global_pointcloud = point_cloud2.read_points(msg, field_names=("x", "y", "z"),skip_nans=True)  # 将PointCloud2数据转换为numpy数组
        point_cloud_list = [point for point in global_pointcloud]
        point_cloud_np = o3d.utility.Vector3dVector(point_cloud_list)
        pc_o3d = o3d.geometry.PointCloud(point_cloud_np)
        self.cloud_ndarray = np.asarray(pc_o3d.points)  # ndarray类型点云数据作为成员对象以供处理
    def img_callback(self, msg):
        global point_2d
        t1 = time.time()
        image_np = np.frombuffer(msg.data, dtype=np.uint8)
        frame = image_np.reshape((msg.height, msg.width, 3))  # frame.shape = (2064, 3088, 3)
        results = self.detector.predict(frame)
        annotator = Annotator(np.ascontiguousarray(frame), line_width=3, example=str(self.detector_second.names))
        if len(results) > 0:
            try:
                point_2d, _ = cv2.projectPoints(self.cloud_ndarray, self.rvec, self.tvec, self.camera_matrix,self.distCoeffs)
            except:
                print("等待稠密点云话题发布中")
            for detection in results:
                label, xywh, confidence = detection
                x, y, w, h = xywh
                x, y, w, h = int(x), int(y), int(w), int(h)
                img2 = frame[y:y + h, x:x + w]
                results_second = self.detector_second.predict(img2)
                if len(results_second) > 0:
                    for detection_second in results_second:
                        label_second, xywh_second, confidence_second = detection_second
                        xyxy_second_in_img1 = []
                        x1_2, y1_2, w_2, h_2 = xywh_second
                        x1 = torch.tensor(x + x1_2, device=self.detector.device)
                        y1 = torch.tensor(y + y1_2, device=self.detector.device)
                        x2 = torch.tensor(x + x1_2 + w_2, device=self.detector.device)
                        y2 = torch.tensor(y + y1_2 + h_2, device=self.detector.device)
                        xyxy_second_in_img1.append(x1)
                        xyxy_second_in_img1.append(y1)
                        xyxy_second_in_img1.append(x2)
                        xyxy_second_in_img1.append(y2)
                    m = -1
                    iou_count = 0
                    mean_cloud = np.float64([0, 0, 0])  # yolo所识别到物体点云平均位置
                    for point in point_2d:
                        m = m + 1
                        if (0 <= m < len(self.cloud_ndarray)) and (x < point[0][0] < x + w) and (y < point[0][1] < y + h):
                            mean_cloud += self.cloud_ndarray[m]
                            iou_count += 1
                    mean_cloud = mean_cloud / iou_count  # 取均值
                    if np.all(np.isnan(mean_cloud)) or iou_count == 0:
                        self.T1 = False
                    annotator.box_label(xyxy_second_in_img1, label_second, color=colors(int(confidence_second), True))
                    if self.enemy_is_blue:
                        if label_second == "B1" and self.T1:
                            self.pos_data[0] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[1] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "B2" and self.T1:
                            self.pos_data[2] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[3] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "B3" and self.T1:
                            self.pos_data[4] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[5] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "B4" and self.T1:
                            self.pos_data[6] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[7] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "B5" and self.T1:
                            self.pos_data[8] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[9] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "B7" and self.T1:
                            self.pos_data[10] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[11] = mean_cloud[1] + self.lidar_pos_y
                            self.guess_7_state = 1
                    else:
                        if label_second == "R1" and self.T1:
                            self.pos_data[0] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[1] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "R2" and self.T1:
                            self.pos_data[2] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[3] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "R3" and self.T1:
                            self.pos_data[4] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[5] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "R4" and self.T1:
                            self.pos_data[6] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[7] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "R5" and self.T1:
                            self.pos_data[8] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[9] = mean_cloud[1] + self.lidar_pos_y
                        elif label_second == "R7" and self.T1:
                            self.pos_data[10] = mean_cloud[0] - self.lidar_pos_x
                            self.pos_data[11] = mean_cloud[1] + self.lidar_pos_y
                            self.guess_7_state = 1
                self.T1 = True
                self.guess_7_decision = self.guess_7_state
                self.guess_7_state = 0
        t2 = time.time()
        print(f"处理一帧耗时为：{t2 - t1}  坐标为：{self.pos_data}")
        img = annotator.result()
        cv2.namedWindow("Detect", 0)
        cv2.imshow("Detect", img)
        cv2.waitKey(1)

    # def timer_callback(self,event):
    #     if self.enemy_is_blue:
    #         t1 = time.time()
    #         if not self.guess_7_decision:
    #             self.pos_data[10] = 22.5
    #             self.pos_data[11] = 7.5
    #         send_interact_blue = com_class.lmap_interaction_send(self.pos_data, 0)
    #         com.send_data(send_interact_blue)
    #         t2 = time.time()
    #         t_wasted = t2 - t1
    #         time.sleep(self.delaytime - t_wasted)
    #     else:
    #         t1 = time.time()
    #         if not self.guess_7_decision:
    #             self.pos_data[10] = 5.7
    #             self.pos_data[11] = 7.5
    #         send_interact_red = com_class.lmap_interaction_send(self.pos_data, 0)
    #         com.send_data(send_interact_red)
    #         t2 = time.time()
    #         t_wasted = t2 - t1
    #         time.sleep(self.delaytime - t_wasted)
    # def timer_callback(self,event):
    #     print("good")

# global com
# com = Port('/dev/ttyUSB0',115200)
# com.openPort()
# global com_class
# com_class = Communicator()
rospy.init_node("lidar_nudt", anonymous=True)
lidar_nudt_1 = lidar_nudt()
rospy.Subscriber("/dense_point_cloud_topic", PointCloud2, lidar_nudt_1.pointcloud_callback)
rospy.Subscriber("image_topic", Image, lidar_nudt_1.img_callback)
rospy.spin()



"""
新战术
1.对于英雄，设置一个定时器，大概前哨战被推掉之前，英雄猜测为定点（大概率在定点打前哨战，那一块点云覆盖不到，且相机视野中也存在遮挡，如果能读取前哨战血量信息最好。
2.对于工程，直接猜定点吧，根据敌方特点一直猜银矿或金矿其中之一的位置，设置一个flag，进行不同的初始化猜点。
3.步兵进行检测。
4.哨兵检测到更新，哨兵出来一般是来打前哨战，“应该”能看清（每遇到这种对手，不知道雷达视角下是什么样），如果检测到就更新，但凡有一帧没有检测到就把哨兵的位置设置为定点。
"""