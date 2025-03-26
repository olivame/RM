#coding=utf-8
import cv2
import numpy as np
# import mvsdk
import platform
#####################
import rospy
import open3d as o3d
import cv2
from pylab import *
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
# import mvsdk
import argparse
import csv
import os
import platform
import sys
from pathlib import Path
import torch
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#########################################！！！！！！通信部分代码！！！！！！#################################################

import serial
import time
import threading
import pdb
import struct
import mmap
import contextlib
from datetime import datetime
import time
import multiprocessing
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
wCRC_Table = [0x0000, 0x1189, 0x2312, 0x329b, 0x4624, 0x57ad, 0x6536, 0x74bf,0x8c48, 0x9dc1, 0xaf5a, 0xbed3, 0xca6c, 0xdbe5, 0xe97e, 0xf8f7,
              0x1081, 0x0108, 0x3393, 0x221a, 0x56a5, 0x472c, 0x75b7, 0x643e,0x9cc9, 0x8d40, 0xbfdb, 0xae52, 0xdaed, 0xcb64, 0xf9ff, 0xe876,
              0x2102, 0x308b, 0x0210, 0x1399, 0x6726, 0x76af, 0x4434, 0x55bd,0xad4a, 0xbcc3, 0x8e58, 0x9fd1, 0xeb6e, 0xfae7, 0xc87c, 0xd9f5,
              0x3183, 0x200a, 0x1291, 0x0318, 0x77a7, 0x662e, 0x54b5, 0x453c,0xbdcb, 0xac42, 0x9ed9, 0x8f50, 0xfbef, 0xea66, 0xd8fd, 0xc974,
              0x4204, 0x538d, 0x6116, 0x709f, 0x0420, 0x15a9, 0x2732, 0x36bb,0xce4c, 0xdfc5, 0xed5e, 0xfcd7, 0x8868, 0x99e1, 0xab7a, 0xbaf3,
              0x5285, 0x430c, 0x7197, 0x601e, 0x14a1, 0x0528, 0x37b3, 0x263a,0xdecd, 0xcf44, 0xfddf, 0xec56, 0x98e9, 0x8960, 0xbbfb, 0xaa72,
              0x6306, 0x728f, 0x4014, 0x519d, 0x2522, 0x34ab, 0x0630, 0x17b9,0xef4e, 0xfec7, 0xcc5c, 0xddd5, 0xa96a, 0xb8e3, 0x8a78, 0x9bf1,
              0x7387, 0x620e, 0x5095, 0x411c, 0x35a3, 0x242a, 0x16b1, 0x0738,0xffcf, 0xee46, 0xdcdd, 0xcd54, 0xb9eb, 0xa862, 0x9af9, 0x8b70,
              0x8408, 0x9581, 0xa71a, 0xb693, 0xc22c, 0xd3a5, 0xe13e, 0xf0b7,0x0840, 0x19c9, 0x2b52, 0x3adb, 0x4e64, 0x5fed, 0x6d76, 0x7cff,
              0x9489, 0x8500, 0xb79b, 0xa612, 0xd2ad, 0xc324, 0xf1bf, 0xe036,0x18c1, 0x0948, 0x3bd3, 0x2a5a, 0x5ee5, 0x4f6c, 0x7df7, 0x6c7e,
              0xa50a, 0xb483, 0x8618, 0x9791, 0xe32e, 0xf2a7, 0xc03c, 0xd1b5,0x2942, 0x38cb, 0x0a50, 0x1bd9, 0x6f66, 0x7eef, 0x4c74, 0x5dfd,
              0xb58b, 0xa402, 0x9699, 0x8710, 0xf3af, 0xe226, 0xd0bd, 0xc134,0x39c3, 0x284a, 0x1ad1, 0x0b58, 0x7fe7, 0x6e6e, 0x5cf5, 0x4d7c,
              0xc60c, 0xd785, 0xe51e, 0xf497, 0x8028, 0x91a1, 0xa33a, 0xb2b3,0x4a44, 0x5bcd, 0x6956, 0x78df, 0x0c60, 0x1de9, 0x2f72, 0x3efb,
              0xd68d, 0xc704, 0xf59f, 0xe416, 0x90a9, 0x8120, 0xb3bb, 0xa232,0x5ac5, 0x4b4c, 0x79d7, 0x685e, 0x1ce1, 0x0d68, 0x3ff3, 0x2e7a,
              0xe70e, 0xf687, 0xc41c, 0xd595, 0xa12a, 0xb0a3, 0x8238, 0x93b1,0x6b46, 0x7acf, 0x4854, 0x59dd, 0x2d62, 0x3ceb, 0x0e70, 0x1ff9,
              0xf78f, 0xe606, 0xd49d, 0xc514, 0xb1ab, 0xa022, 0x92b9, 0x8330,0x7bc7, 0x6a4e, 0x58d5, 0x495c, 0x3de3, 0x2c6a, 0x1ef1, 0x0f78]

class Port:
    def __init__(self, port, band):
        self.port = port
        self.baud = int(band)
        self.__open_port = None
        self.get_data_flag = True
        self.__real_time_all_data = b''    #接受的所有数据
        self.__real_time_one_data = b''    #一次接受的数据

    def get_real_time_all_data(self):
        ret = self.__real_time_all_data
        self.__real_time_all_data = b''
        return ret
    def get_real_time_one_data(self):
        return self.__real_time_one_data
    def clear_real_time_data(self):
        self.__real_time_all_data = b''
    #设置是否接收数据
    def set_get_data_flag(self, get_data_flag):
        self.get_data_flag = get_data_flag

    def openPort(self):
        try:
            self.__open_port = serial.Serial(self.port, self.baud)
            threading.Thread(target=self.get_data, args=()).start()   #开启数据接收线程
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
                    data = self.__open_port.read()  #一次读所有的字节
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

	def lmap_interaction_send(self, robot_ID, robot_x, robot_y, robot_toward, seq):  # 小地图交互
		ID = int(robot_ID).to_bytes(2, self.storage_mode)
		# x = int(robot_x).to_bytes(4,self.storage_mode,signed=True)
		# y = int(robot_y).to_bytes(4,self.storage_mode,signed=True)
		x = struct.pack('f', robot_x)
		y = struct.pack('f', robot_y)
		toward = struct.pack('f', robot_toward)
		data = ID + x + y + toward
		cmd_id = int(0x0305).to_bytes(2, self.storage_mode)
		return self.package_data(cmd_id, seq, data)
		"""
    operate_tpye  0:空操作 
                  1:删除图层
                  2: 删除所有
    layer：       0~9
        """

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

	def read_time(self, data):
		time_left = int.from_bytes(data, 'little')
		return time_left

	def read_blood(self, data):
		ROBOT_BLOODS = []
		blood_R1 = int.from_bytes(data[0:2], 'little')
		ROBOT_BLOODS.append(blood_R1)
		# print('红1英雄机器人血量为',blood_R1)
		blood_R2 = int.from_bytes(data[2:4], 'little')
		ROBOT_BLOODS.append(blood_R2)
		# print('红2工程机器人血量为',blood_R2)
		blood_R3 = int.from_bytes(data[4:6], 'little')
		ROBOT_BLOODS.append(blood_R3)
		# print('红3步兵机器人血量为',blood_R3)
		blood_R4 = int.from_bytes(data[6:8], 'little')
		ROBOT_BLOODS.append(blood_R4)
		# print('红4步兵机器人血量为',blood_R4)
		blood_R5 = int.from_bytes(data[8:10], 'little')
		ROBOT_BLOODS.append(blood_R5)
		# print('红5步兵机器人血量为',blood_R5)
		blood_R7 = int.from_bytes(data[10:12], 'little')
		ROBOT_BLOODS.append(blood_R7)
		# print('红7哨兵机器人血量为',blood_R7)
		blood_R_POST = int.from_bytes(data[12:14], 'little')
		ROBOT_BLOODS.append(blood_R_POST)
		# print('红方前哨站血量为',blood_R_POST)
		blood_R_BASE = int.from_bytes(data[14:16], 'little')
		ROBOT_BLOODS.append(blood_R_BASE)
		# print('红方基地血量为',blood_R_BASE)
		blood_B1 = int.from_bytes(data[16:18], 'little')
		ROBOT_BLOODS.append(blood_B1)
		# print('蓝1英雄机器人血量为',blood_B1)
		blood_B2 = int.from_bytes(data[18:20], 'little')
		ROBOT_BLOODS.append(blood_B2)
		# print('蓝2工程机器人血量为',blood_B2)
		blood_B3 = int.from_bytes(data[20:22], 'little')
		ROBOT_BLOODS.append(blood_B3)
		# print('蓝3步兵机器人血量为',blood_B3)
		blood_B4 = int.from_bytes(data[22:24], 'little')
		ROBOT_BLOODS.append(blood_B4)
		# print('蓝4步兵机器人血量为',blood_B4)
		blood_B5 = int.from_bytes(data[24:26], 'little')
		ROBOT_BLOODS.append(blood_B5)
		# print('蓝5步兵机器人血量为',blood_B5)
		blood_B7 = int.from_bytes(data[26:28], 'little')
		ROBOT_BLOODS.append(blood_B7)
		# print('蓝7哨兵机器人血量为',blood_B7)
		blood_B_POST = int.from_bytes(data[28:30], 'little')
		ROBOT_BLOODS.append(blood_B_POST)
		# print('蓝方前哨站血量为',blood_B_POST)
		blood_B_BASE = int.from_bytes(data[30:32], 'little')
		ROBOT_BLOODS.append(blood_B_BASE)
		return ROBOT_BLOODS

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
			# if cmd_id == self.cmd_id_blood:
			#     print("血量信息")
			#     data_blood = data_pack[7:35]
			#     self.read_blood(data_blood)
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
				mark1 = int.from_bytes(data_pack[7:8], 'little')
				mark2 = int.from_bytes(data_pack[8:9], 'little')
				mark3 = int.from_bytes(data_pack[9:10], 'little')
				mark4 = int.from_bytes(data_pack[11:12], 'little')
				mark5 = int.from_bytes(data_pack[12:13], 'little')
				mark6 = int.from_bytes(data_pack[13:14], 'little')
				mark_all = [mark1, mark2, mark3, mark4, mark5, mark6]
				return mark_all

#
# def send_position(q_track_identities, q_isblue, q_point2d, com, com_class, enemy, stable_mode):
# 	# 以左下角为坐标原点进行发送xy位置信息，水平向右为x，竖直向上为y，大小为28*15
# 	# 云台手小地图q_point2d左上角为原点，水平向右为x，竖直向下为y，长为map_w：279，宽为map_h：149
# 	global isblue
# 	global track_identities
# 	global point2d
# 	# stable_mode=0   #是否为固定发定位点模式
# 	x_length = 28
# 	y_length = 15
# 	isblue = []
# 	track_identities = []
# 	point2d = [1, 1]
# 	# 没有3
# 	track_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 3, 9: 101, 10: 102, 11: 103, 12: 104, 13: 105, 14: 106,
# 				15: 107, 16: 103}
# 	delaytime = 0.1
# 	t_move = 0.1
# 	red_id = [1, 2, 3, 4, 5, 6, 7, 8]
# 	blue_id = [9, 10, 11, 12, 13, 14, 15, 16]
# 	time.sleep(2)
# 	judge_isblue = 2
# 	mark_past_all = [0, 0, 0, 0, 0, 0]
# 	while 1:
# 		# mark_info = q_mark.get()
# 		# print('接受到的标记为',mark_info)
# 		# if not q_xyxys.empty():
# 		#     track_xyxys = q_xyxys.get()
# 		# print("串口中track_xyxys为", track_xyxys)
# 		# print("读取消息队列结果")
# 		if not q_track_identities.empty():
# 			id_send = q_track_identities.get()
# 		if not q_isblue.empty():
# 			judge_isblue = q_isblue.get()
# 		if not q_point2d.empty():
# 			point2d = q_point2d.get()
# 			print("消息队列读取的结果：颜色", judge_isblue, '编号', id_send, '位置', point2d[0] / 279 * x_length,
# 				  y_length - point2d[1] / 149 * y_length)
#
# 		# send_interact = com_class.lmap_interaction_send(7, 13.5, 3.5, 0.1, 0)
# 		# com.send_data(send_interact)
# 		# time.sleep(delaytime)
# 		# send_interact = com_class.lmap_interaction_send(1, 14, 3, 0.1, 0)
# 		# com.send_data(send_interact)
# 		# time.sleep(delaytime)
#
# 		# 还没识别到则跳过
# 		if (judge_isblue == 2):
# 			time.sleep(0.2)
# 			continue
# 		try:
# 			# print('数量为',len(track_xyxys),'或',len(judge_isblue),'判断颜色为',judge_isblue)
# 			#     send_interact = com_class.lmap_interaction_send(7, 15, 5, 0.1, 0)
# 			#     com.send_data(send_interact)
#
# 			time.sleep(delaytime)
# 			######################################################我方为红方，敌方为蓝方##########################################
# 			if (enemy == 'blue'):
# 				##########################定点发送##########################################(19.15,12.6)19.2, 12.8
# 				if (stable_mode == 1):
# 					send_stable_b1 = com_class.lmap_interaction_send(101, 17.2, 4.6, 0.1, 0)  # 17,1.3,
# 					com.send_data(send_stable_b1)
# 					time.sleep(delaytime)
# 					send_stable_b2 = com_class.lmap_interaction_send(102, 9.643, 10.775, 0.1, 0)  # 工程
# 					com.send_data(send_stable_b1)
# 					time.sleep(delaytime)
# 					send_stable_b3 = com_class.lmap_interaction_send(103, 28, 15, 0.1, 0)
# 					com.send_data(send_stable_b3)
# 					time.sleep(delaytime)
# 					send_stable_b4 = com_class.lmap_interaction_send(104, 19.5, 13.5, 0.1, 0)  # 15.5, 12.5
# 					com.send_data(send_stable_b4)
# 					time.sleep(delaytime)
# 					send_stable_b5 = com_class.lmap_interaction_send(105, 19.5, 13.5, 0.1, 0)
# 					com.send_data(send_stable_b5)
# 					time.sleep(delaytime)
# 					send_stable_b6 = com_class.lmap_interaction_send(106, 26.8, 1.2, 0.1, 0)
# 					com.send_data(send_stable_b6)
# 					time.sleep(delaytime)
#
# 				#################################根据识别发送########################
# 				if (stable_mode == 0):
# 					# id_send = track_identities[k]
# 					if (judge_isblue == 0):  # 如果识别为己方红方则跳过发送位置
# 						time.sleep(0.001)
# 						continue
# 					if (id_send > 16):
# 						id_send = blue_id[id_send % 8]
# 					if (id_send <= 8):
# 						id_send = blue_id[id_send % 8]
# 					send_interact2 = com_class.lmap_interaction_send(102, 9.643, 10.775, 0.1, 0)
# 					com.send_data(send_interact2)
# 					time.sleep(delaytime)
#
# 					#send_interact = com_class.lmap_interaction_send(track_id[id_send], point2d[0]  - 1.628,
# 																	#y_length + point2d[1]  - 5.59, 0.1, 0)
# 					send_interact = com_class.lmap_interaction_send(track_id[id_send], point2d[0] ,
# 																	y_length + point2d[1] - 1.30, 0.1, 0)
# 					'''
# 					send_interact = com_class.lmap_interaction_send(track_id[id_send],
# 																	point2d[0] / 279 * x_length ,
# 																	y_length + point2d[1] / 149 * y_length , 0.1,
# 																	0)
# 					'''
# 					# if ((abs(point2d[0] / 279 * 30 - 6.6) + abs( 20 - point2d[1] / 149 * 20 - 8.5)) < 2):  # 识别位置在哨兵附近则忽略
# 					#     continue
# 					com.send_data(send_interact)
# 					print(" 要发送的ID为", id_send, track_id[id_send], "位置为",
# 						  point2d[0] / 279 * x_length - 1.628,
# 						  y_length + point2d[1] / 149 * y_length - 5.59)
# 					time.sleep(delaytime)
#
# 				# for i in range(len(mark_info)):
# 				#     mark_i_now=mark_info[i]
# 				#     if(mark_i_now>mark_past_all[i]):    #成功标记
# 				#         while(1):   #初步重复发该位置，进一步：利用两者信息
# 				#             com.send_data(send_interact)
# 				#             mark_past_all[i]=mark_i_now
# 				#             mark_info = q_mark.get()
# 				#             mark_i_now = mark_info[i]
# 				#             time.sleep(delaytime)
# 				#             if (mark_i_now <= mark_past_all[i]):
# 				#                 break
# 			################################################################我方为蓝方，敌方为红方##########################################
# 			if (enemy == 'red'):
# 				##########################定点发送##########################################
# 				if (stable_mode == 1):
# 					send_stable_r1 = com_class.lmap_interaction_send(1, 11, 13.7, 0.1, 0)
# 					com.send_data(send_stable_r1)
# 					time.sleep(delaytime)
# 					send_stable_r2 = com_class.lmap_interaction_send(2, 18.357, 4.225, 0.1, 0)
# 					com.send_data(send_stable_r1)
# 					time.sleep(delaytime)
# 					send_stable_r3 = com_class.lmap_interaction_send(3, 8.5, 1.5, 0.1, 0)
# 					com.send_data(send_stable_r3)
# 					time.sleep(delaytime)
# 					send_stable_r4 = com_class.lmap_interaction_send(4, 8.5, 1.5, 0.1, 0)
# 					com.send_data(send_stable_r4)
# 					time.sleep(delaytime)
# 					send_stable_r5 = com_class.lmap_interaction_send(5, 8.5, 1.5, 0.1, 0)
# 					com.send_data(send_stable_r5)
# 					time.sleep(delaytime)
# 					send_stable_r6 = com_class.lmap_interaction_send(6, 1.2, 13.8, 0.1, 0)
# 					com.send_data(send_stable_r6)
# 					time.sleep(delaytime)
#
# 				#################################根据识别发送########################
# 				if (stable_mode == 0):
# 					# id_sends = track_identities[k]
# 					if (judge_isblue == 1):  # 如果识别为己方蓝方则跳过发送位置
# 						time.sleep(0.001)
# 						continue
# 					if (id_send > 8):
# 						id_send = red_id[id_send % 8]
# 					send_interact2 = com_class.lmap_interaction_send(2, 18.357, 4.225, 0.1, 0)
# 					com.send_data(send_interact2)
# 					time.sleep(delaytime)
# 					send_interact = com_class.lmap_interaction_send(track_id[id_send],
# 																	x_length - point2d[0] / 279 * x_length + 1.628,
# 																	5.59 - point2d[1] / 149 * y_length, 0.1, 0)
#
# 					# if ((abs(point2d[0] / 279 * 30 - 6.6) + abs( 20 - point2d[1] / 149 * 20 - 8.5)) < 2):  # 识别位置在哨兵附近则忽略
# 					#     continue
# 					com.send_data(send_interact)
# 					print("要发送的ID为", id_send, track_id[id_send], "位置为",
# 						  x_length - point2d[0] / 279 * x_length + 1.628,
# 						  5.59 - point2d[1] / 149 * y_length,)
# 					time.sleep(delaytime)
# 				# for i in range(len(mark_info)):
# 				#     mark_i_now=mark_info[i]
# 				#     if(mark_i_now>mark_past_all[i]):    #成功标记
# 				#         while(1):   #初步重复发该位置，进一步：利用两者信息
# 				#             com.send_data(send_interact)
# 				#             mark_past_all[i]=mark_i_now
# 				#             mark_info = q_mark.get()
# 				#             mark_i_now = mark_info[i]
# 				#             time.sleep(delaytime)
# 				#             if (mark_i_now <= mark_past_all[i]):
# 				#                 break
#
# 		# if (enemy == 'blue'):   #自己为红方则自己在左半面
# 		#     send_interact = com_class.lmap_interaction_send(track_id[id_send], point2d[0]/279*30, 20 - point2d[1]/149*20, 0.1, 0)
# 		# else:   ##自己为蓝方则自己在右半面
# 		#     send_interact = com_class.lmap_interaction_send(track_id[id_send], 30-point2d[0] / 279 * 30,point2d[1] / 149 * 20, 0.1, 0)
# 		# com.send_data(send_interact)
#
# 		# print("距离结束还有" + str(time_left) + "秒:要发送的ID为", id_send, track_id[id_send], "位置为", point2d[0] / 279 * 30, 20 - point2d[1] / 149 * 20)
# 		# send_str="距离结束还有"+str(time_left)+"秒：要发送的ID为"+str(track_id[id_send])+"位置为"+str(point2d[0]/279*30)+"," +str(20 - point2d[1]/149*20)
# 		# record_file.write(send_str+"\n")
# 		# time.sleep(delaytime)
# 		except Exception as e:
# 			continue
########################################！！！！！！以上为通信部分代码！！！！！！#############################################################################








####################################！！！！！！以下为检测部分！！！！！！##########################################################################################
#相机图像去畸变
"""
# 第一版去畸变函数
def undistort(k,d,frame):

	# k=np.array( [[ 1946.5 , 0 , 1586.6],
    #             [  0 , 1943.3 , 999.5301],apt autoremove
    #             [  0 , 0,  1]])

	# d=np.array([-0.4368 , 0.2516 ,-0.00092121 ,-0.00034122 ,-0.0825  ])

	h,w=frame.shape[:2]
	#k是内参,d是畸变矩阵,R两个摄像头时才考虑,
	mapx,mapy=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),5)
	return cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
"""

class yolo_myself:
	def __init__(self):
		self.lidar_pos_x=1.628
		self.lidar_pos_y=9.41
		self.position_blue0 = [17.2, 4.6]	#敌方蓝英雄
		self.position_blue1 = [18.4, 4.2]	#工程
		self.position_blue2 = [19.5, 13.5]	#步兵
		self.position_blue3 = [19.5, 13.5]	#步兵
		self.position_blue4 = [19.5, 13.5]	#步兵
		self.position_blue5 = [22.5, 7.5]	#哨兵
		self.position_red0 = [11, 13.7]		#敌方红英雄
		self.position_red1 = [9.6, 10.8]	#工程
		self.position_red2 = [8.5, 1.5]		#步兵
		self.position_red3 = [8.5, 1.5]		#步兵
		self.position_red4 = [8.5, 1.5]		#步兵
		self.position_red5 = [5.7, 7.5]		#哨兵

		self.track_identities_blue0 = 101
		self.track_identities_blue1 = 102
		self.track_identities_blue2 = 103
		self.track_identities_blue3 = 104
		self.track_identities_blue4 = 105
		self.track_identities_blue5 = 107
		self.track_identities_red0 = 1
		self.track_identities_red1 = 2
		self.track_identities_red2 = 3
		self.track_identities_red3 = 4
		self.track_identities_red4 = 5
		self.track_identities_red5 = 7

		# self.track_id = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 3, 9: 101, 10: 102, 11: 103, 12: 104, 13: 105, 14: 106, 15: 107, 16: 103}
		self.x_length = 28
		self.y_length = 15
		self.delaytime = 0.1

		self.enemy_is_blue = True
		self.enemy_is_red = False

		self.armor_pos_in_im0=[0,0,1,1]  # 不随便初始化一个回报错，不懂为啥

		self.send_list_blue = [0,2,3]  # 选择发送车辆的标号
		self.send_list_red = [6,8,9]



		weight_first = "/home/ldk/RM_lidar/yolov5_camera_final_1/weights_test/r1/best.pt"
		weight_second = "/home/ldk/RM_lidar/yolov5_camera_final_1/weights_test/r2/best.pt"
		data_first = "/home/ldk/yolov5-master/data/coco128.yaml"#检测好像用不到，我就瞎写了一个
		data_second = "/home/ldk/yolov5-master/data/coco128.yaml"
		dnn = False  # use OpenCV DNN for ONNX inference

		imgsz = (3088, 2064)
		bs = 1  # batch_size

		self.bridge = CvBridge()
		self.conf_thres = 0.2
		self.iou_thres = 0.45
		#加载第一阶段模型
		device=select_device()
		self.model_first=DetectMultiBackend(weight_first,device=device,dnn=dnn,data=data_first,fp16=False)
		self.stride_first, self.names_first, self.pt_first = self.model_first.stride, self.model_first.names, self.model_first.pt
		self.imgsz_first = check_img_size(imgsz, s=self.stride_first)
		#加载第二阶段模型
		self.model_second = DetectMultiBackend(weight_second, device=device, dnn=dnn, data=data_second, fp16=False)
		self.stride_second, self.names_second, self.pt_second = self.model_second.stride, self.model_second.names, self.model_second.pt
		self.imgsz_second = check_img_size(imgsz, s=self.stride_second)
		# Run inference(预热)
		self.model_first.warmup(imgsz=(1 if self.pt_first or self.model_first.triton else bs, 3, *self.imgsz_first))
		self.model_second.warmup(imgsz=(1 if self.pt_second or self.model_second.triton else bs, 3, *self.imgsz_second))
		self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())

		self.timer = rospy.Timer(rospy.Duration(0.3), self.timer_callback)
		# 判断条件
		self.T1=False #第二阶段是否检测到车，默认没有(False),有的话改为True
		self.T2=False#第二阶段是否检测到车，默认没有(False),有的话改为True,这个用于绘制装甲板，计算点云版
		self.T3 = False  # 第二阶段是否检测到车，默认没有(False),有的话改为True,这个用于绘制装甲板

		# 雷达初始化
		# rospy.init_node("yolo_myself", anonymous=True)
		# self.pointcloud_callback -> 将点云消息转化为ndarray类型点云数据以进行处理
		# self.pointcloud_sub = rospy.Subscriber("/dense_point_cloud_topic", PointCloud2, self.pointcloud_callback)

		# 旋转平移矩阵不准确，需要修改
		# 经过矩阵转置，以及罗德里格斯变换得到的旋转矩阵    HAP
		# 第一版：现采用谢旭辉班长标定出,左相机
		# self.Rotate_matrix = np.float64([[0.132624, -0.99113, -0.008442],
		# 								 [0.0110486, -0.00703852, -0.999914],
		# 								 [0.991105, 0.132706, 0.010017]])
		# self.rvec, _ = cv2.Rodrigues(self.Rotate_matrix)
		#
		#
		# # 经过排序修改后得到的平移矩阵
		# self.tvec = np.float64([0.179374, -0.126138, -0.53179])
		#
		# # 相机内部参数         matlab
		# # 第一版：相机内参采用谢旭辉班长matlab标定结果
		# self.camera_matrix = np.float64([[1922.3, 0, 1580],
		# 								 [0, 1923, 1023.6],
		# 								 [0, 0, 1]])
		# # # 相机形变矩阵
		# # # self.distCoeffs = np.float64([-0.42427578,0.25358,-0.0004371654,-0.00020451,-0.096708])
		# self.distCoeffs = np.float64([-0.4551,0.2824,-0.0017,0.0026,-0.1042])

		# 右相机
		self.Rotate_matrix = np.float64([[-0.195049, -0.980605, -0.0192303],
										 [0.0239639, 0.0148363, -0.999603],
										 [0.980501, -0.195432, 0.0206054]])
		self.rvec, _ = cv2.Rodrigues(self.Rotate_matrix)
		# 经过排序修改后得到的平移矩阵
		self.tvec = np.float64([-0.16477, -0.0466339, 0.052998])

		# 相机内部参数         matlab
		# 第一版：相机内参采用谢旭辉班长matlab标定结果
		self.camera_matrix = np.float64([[1947,0,1576.9],
										 [0,1946.9,1061.2],
										 [0, 0, 1]])
		# # 相机形变矩阵
		# # self.distCoeffs = np.float64([-0.42427578,0.25358,-0.0004371654,-0.00020451,-0.096708])
		self.distCoeffs = np.float64([-0.4509,0.2993,-0.00009985,0.0001312,-0.1297])

		self.output = []
		#print("模型加载+预热+参数初始化完毕")




	# def SolvePosition(self,xyxy):
	#
	# 	# 进行点云由3D到2D的转换
	# 		try:
	# 			point_2d, _ = cv2.projectPoints(self.cloud_ndarray, self.rvec, self.tvec, self.camera_matrix, self.distCoeffs)
	# 		except:
	# 			print("等待稠密点云发布中")
	# 			return
	#
	# 		x = []
	# 		y = []
	# 		distance = []
	# 		m = -1
	# 		im_w, im_h=self.im.shape[1], self.im.shape[0]
	# 		iou_count = 0
	# 		mean_cloud = np.float64([0, 0, 0]) #yolo所识别到物体点云平均位置
	# 		for point in point_2d:
	# 			m = m+1
	# 			x_2d = point[0][0]
	# 			y_2d = point[0][1]
	#
	# 			if (0 <= x_2d <=im_w) and (0 <= y_2d <= im_h) and (0 <= m < len(self.cloud_ndarray)):
	# 				x.append(x_2d)
	# 				y.append(y_2d)
	# 				distance.append((self.cloud_ndarray[m,0]**2 + self.cloud_ndarray[m,1]**2 + self.cloud_ndarray[m,2]**2)**0.5)
	#
	# 			if (0 <= m <= len(self.cloud_ndarray)) and (xyxy[0] < x_2d <xyxy[2]) and (xyxy[1] < y_2d <xyxy[3]):
	# 				mean_cloud+=self.cloud_ndarray[m]
	# 				iou_count+=1
	# 		mean_cloud=mean_cloud/iou_count #取均值
			# 减少算力时考虑删去

			# try:
			# 	color_pp = int(max(distance) - min(distance))
			# except:
			# 	color_pp = 1000000
			# for i in range(len(x)):
			# 	x_2d = x[i]
			# 	y_2d = y[i]
			# 	try:
			# 		color = int((distance[i] * 255)/color_pp)
			# 	except:
			# 		color = 0
			# 	cv2.circle(self.im, (int(x_2d), int(y_2d)), 1, (0, 0, color), -1)
			# cv2.namedWindow("Image with Points", 0)
			# cv2.imshow("Image with Points", self.im)
			# cv2.waitKey(1)

			# return mean_cloud

	# 将点云消息转化为ndarray类型点云数据以进行处理
	def pointcloud_callback(self,msg):
		#将点云话题转换成open3d可以处理的类型
		global_pointcloud = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True) # 将PointCloud2数据转换为numpy数组
		point_cloud_list = [point for point in global_pointcloud]
		point_cloud_np = o3d.utility.Vector3dVector(point_cloud_list)
		pc_o3d = o3d.geometry.PointCloud(point_cloud_np)
		self.cloud_ndarray = np.asarray(pc_o3d.points)  # ndarray类型点云数据作为成员对象以供处理

	# 输入网络层前预处理图像数据
	def preprocess_first(self, img):
		"""
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
		img0 = img.copy()
		img = np.array([letterbox(img, self.imgsz_first, stride=self.stride_first, auto=self.pt_first)[0]])
		# Convert
		img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
		img = np.ascontiguousarray(img)
		return img, img0

	def preprocess_second(self, img):
		"""
	    Adapted from yolov5/utils/datasets.py LoadStreams class
	    """
		img0 = img.copy()
		img = np.array([letterbox(img, self.imgsz_second, stride=self.stride_second, auto=self.pt_second)[0]])
		# Convert
		img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
		img = np.ascontiguousarray(img)
		return img, img0

	def timer_callback(self,event):

		#ta = time.time()
		if self.enemy_is_blue:
			send_interact_blue0 = com_class.lmap_interaction_send(self.track_identities_blue0,self.position_blue0[0],self.position_blue0[1], 0.1, 0)
			com.send_data(send_interact_blue0)
			time.sleep(self.delaytime)

			# send_interact_blue1= com_class.lmap_interaction_send(self.track_identities_blue1,self.position_blue1[0],self.position_blue1[1], 0.1, 0)
			# com.send_data(send_interact_blue1)
			# time.sleep(self.delaytime)

			send_interact_blue2= com_class.lmap_interaction_send(self.track_identities_blue2,self.position_blue2[0],self.position_blue2[1], 0.1, 0)
			com.send_data(send_interact_blue2)
			time.sleep(self.delaytime)

			send_interact_blue3 = com_class.lmap_interaction_send(self.track_identities_blue3, self.position_blue3[0],self.position_blue3[1], 0.1, 0)
			com.send_data(send_interact_blue3)
			time.sleep(self.delaytime)

			# send_interact_blue4 = com_class.lmap_interaction_send(self.track_identities_blue4, self.position_blue4[0], self.position_blue4[1], 0.1, 0)
			# com.send_data(send_interact_blue4)
			# time.sleep(self.delaytime)
			#
			# send_interact_blue5 = com_class.lmap_interaction_send(self.track_identities_blue5, self.position_blue5[0], self.position_blue5[1], 0.1, 0)
			# com.send_data(send_interact_blue5)
			# time.sleep(self.delaytime)
			#print("已发送")

		elif self.enemy_is_red:
			send_interact_red0 = com_class.lmap_interaction_send(self.track_identities_red0, self.position_red0[0],self.position_red0[1], 0.1, 0)
			com.send_data(send_interact_red0)
			time.sleep(self.delaytime)

			# send_interact_red1 = com_class.lmap_interaction_send(self.track_identities_red1, self.position_red1[0],self.position_red1[1], 0.1, 0)
			# com.send_data(send_interact_red1)
			# time.sleep(self.delaytime)

			send_interact_red2 = com_class.lmap_interaction_send(self.track_identities_red2, self.position_red2[0],self.position_red2[1], 0.1, 0)
			com.send_data(send_interact_red2)
			time.sleep(self.delaytime)

			send_interact_red3 = com_class.lmap_interaction_send(self.track_identities_red3, self.position_red3[0],self.position_red3[1], 0.1, 0)
			com.send_data(send_interact_red3)
			time.sleep(self.delaytime)

			# send_interact_red4 = com_class.lmap_interaction_send(self.track_identities_red4, self.position_red4[0],self.position_red4[1], 0.1, 0)
			# com.send_data(send_interact_red4)
			# time.sleep(self.delaytime)
			#
			# send_interact_red5 = com_class.lmap_interaction_send(self.track_identities_red5, self.position_red5[0],self.position_red5[1], 0.1, 0)
			# com.send_data(send_interact_red5)
			# time.sleep(self.delaytime)

			#print("已发送")
		#tb = time.time()
		#print("timer一次耗时：",tb-ta)


	def yolt_callback(self,data):
		#雷达定位
		# 李树程2
		#self.im = self.bridge.imgmsg_to_cv2(data,desired_encoding="bgr8") #这行还需要修改
		self.im=data
		im,im0=self.preprocess_first(self.im)
		with self.dt[0]:
			im = torch.from_numpy(im).to(self.model_first.device)
			im = im.half() if self.model_first.fp16 else im.float()  # uint8 to fp16/32
			im /= 255  # 0 - 255 to 0.0 - 1.0
			if len(im.shape) == 3:
				im = im[None]  # expand for batch dim
		with self.dt[1]:
			pred_first=self.model_first(im,augment=False,visualize=False)
		with self.dt[2]:
			pred_first=non_max_suppression(pred_first, self.conf_thres, self.iou_thres, None, False, max_det=5) #这行还需要修改

		for i,det in enumerate(pred_first):
			annotator_first =Annotator(im0,line_width=3,example=str(self.names_first))
			if len(det):
				try:
					point_2d, _ = cv2.projectPoints(self.cloud_ndarray, self.rvec, self.tvec, self.camera_matrix,self.distCoeffs)
				except:
					print("等待稠密点云发布中")
					break
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
				#Write results
				for *xyxy, conf, cls in reversed(det):
					#t0 = time.time()
					c = int(cls)
					label_first = f"{self.names_first[c]}"
					confidence_first = float(conf)
					confidence_str_first = f"{confidence_first:.2f}"
					#这行还需要修改#绘图
					#开始第二阶段检测
					#取出车在第一阶段图像中像素坐标
					pos_in_im0=[]
					for i in range(4):
						pos_in_im0.append(xyxy[i].item())
					self.im_second=im0[int(pos_in_im0[1]):int(pos_in_im0[3]),int(pos_in_im0[0]):int(pos_in_im0[2])]#先y后x
					im_second,im_second_0=self.preprocess_second(self.im_second)
					im_second=torch.from_numpy(im_second).to(self.model_second.device)
					im_second = im_second.half() if self.model_second.fp16 else im_second.float()
					im_second /=255
					if len(im_second.shape) == 3:
						im_second=im_second[None]
					pred_second=self.model_second(im_second,augment=False,visualize=False)
					pred_second=non_max_suppression(pred_second,self.conf_thres, self.iou_thres, None, False, max_det=5)
					#c_second_list=[]
					#confidence_second_list=[]
					#confidence_str_second_list=[]
					for j,det_second in enumerate(pred_second):
						if (len(det_second)==0):
							pass
							#print("第二阶段没有识别出具体兵种\n")
						if len(det_second):#0为假
							#self.T1=True
							#self.T2=True
							self.T3 = True
							det_second[:,:4] = scale_boxes(im_second.shape[2:],det_second[:,:4],im_second_0.shape).round()
							for *xyxy_second ,conf_second ,cls_second in reversed(det_second):
								c=int(cls_second)
								label_first = f"{self.names_second[c]}"
								confidence_first = float(conf_second)
								confidence_str_first = f"{confidence_first:.2f}"
								pos_in_im_second_0=[]
								for i in range(4):
									pos_in_im_second_0.append(xyxy_second[i].item())
								self.armor_pos_in_im0=[pos_in_im0[0]+pos_in_im_second_0[0],
												  pos_in_im0[1]+pos_in_im_second_0[1],
												  pos_in_im0[0]+pos_in_im_second_0[2],
												  pos_in_im0[1]+pos_in_im_second_0[3]]
					if self.T3:
						xyxy_np = [float((i.cpu()).numpy()) for i in xyxy]  # Tensor-gpu转cpu再转np
						output = []
						if (c in self.send_list_blue) and self.enemy_is_blue:
							x = []
							y = []
							# distance = []
							m = -1
							im_w, im_h = self.im.shape[1], self.im.shape[0]
							iou_count = 0
							mean_cloud = np.float64([0, 0, 0])  # yolo所识别到物体点云平均位置
							for point in point_2d:
								m =m+1
								x_2d = point[0][0]
								y_2d = point[0][1]

								if (0 <= x_2d <= im_w) and (0 <= y_2d <= im_h) and (0 <= m < len(self.cloud_ndarray)):
									x.append(x_2d)
									y.append(y_2d)
									# distance.append((self.cloud_ndarray[m, 0] ** 2 + self.cloud_ndarray[m, 1] ** 2 +self.cloud_ndarray[m, 2] ** 2) ** 0.5)
								if (0 <= m <= len(self.cloud_ndarray)) and (xyxy[0] < x_2d < xyxy[2]) and (xyxy[1] < y_2d < xyxy[3]):
									mean_cloud += self.cloud_ndarray[m]
									iou_count += 1
							mean_cloud = mean_cloud / iou_count  # 取均值
							# mean_cloud = self.SolvePosition(xyxy_np)
							print(f"{label_first}的位置为：{mean_cloud}")
							for i in range(0, 2):
								output.append(mean_cloud[i])
							output.append(c)
							self.output = output
						elif (c in self.send_list_red) and self.enemy_is_red:
							x = []
							y = []
							# distance = []
							m = -1
							im_w, im_h = self.im.shape[1], self.im.shape[0]
							iou_count = 0
							mean_cloud = np.float64([0, 0, 0])  # yolo所识别到物体点云平均位置
							for point in point_2d:
								m = m + 1
								x_2d = point[0][0]
								y_2d = point[0][1]

								if (0 <= x_2d <= im_w) and (0 <= y_2d <= im_h) and (0 <= m < len(self.cloud_ndarray)):
									x.append(x_2d)
									y.append(y_2d)
								# distance.append((self.cloud_ndarray[m, 0] ** 2 + self.cloud_ndarray[m, 1] ** 2 +self.cloud_ndarray[m, 2] ** 2) ** 0.5)
								if (0 <= m <= len(self.cloud_ndarray)) and (xyxy[0] < x_2d < xyxy[2]) and (
										xyxy[1] < y_2d < xyxy[3]):
									mean_cloud += self.cloud_ndarray[m]
									iou_count += 1
							mean_cloud = mean_cloud / iou_count  # 取均值
							print(f"{label_first}的位置为：{mean_cloud}")
							for i in range(0, 2):
								output.append(mean_cloud[i])
							output.append(c)
							self.output = output
						else:
							self.output = []
						"""
						output = []
						for i in range(0, 2):
							output.append(mean_cloud[i])
						output.append(c)
						self.output = output
						"""
						xyxy_second_in_im0 = []
						for i in range(4):
							a = self.armor_pos_in_im0[i]
							a = torch.tensor(a, device=self.model_first.device)
							xyxy_second_in_im0.append(a)
						annotator_first.box_label(xyxy_second_in_im0, label_first + " " + confidence_str_first,color=colors(c, True))

					"""
					else:
						xyxy_np = [float((i.cpu()).numpy()) for i in xyxy]  # Tensor-gpu转cpu再转np
						if (0<= c <=5):
						#if (6<= c <=11):
							mean_cloud = self.SolvePosition(xyxy_np)
						else:
							mean_cloud = [0,0,0]
						output = []
						for i in range(0, 2):
							output.append(mean_cloud[i])
						output.append(c)101101
						self.output = output
						annotator_first.box_label(xyxy, label_first + " " + confidence_str_first, color=colors(c, True))
					"""


					if (len(self.output) > 0) and (self.output[2] in self.send_list_blue) and self.T3:
						point2d = self.output[0:2]  # 位置
						point2d[0]=point2d[0]-self.lidar_pos_x
						point2d[1]=point2d[1]+self.lidar_pos_y
						# 蓝方
						# 英雄1号
						if self.output[2] == 0:
							#坐标转化有待处理
							self.position_blue0 = point2d
							self.track_identities_blue0 = 101
							# send_interact_blue0 = com_class.lmap_interaction_send(self.track_identities_blue0,self.position_blue0[0],self.position_blue0[1], 0.1, 0)
							# com.send_data(send_interact_blue0)

						# elif self.output[2] == 1:
						# 	self.position_blue1 = point2d
						# 	self.track_identities_blue1 = 102
							# send_interact_blue1 = com_class.lmap_interaction_send(self.track_identities_blue1,self.position_blue1[0],self.position_blue1[1], 0.1, 0)
							# com.send_data(send_interact_blue1)

						# 步兵3号
						elif self.output[2] == 2:
							self.position_blue2 = point2d
							self.track_identities_blue2 = 103
							# send_interact_blue2 = com_class.lmap_interaction_send(self.track_identities_blue2,self.position_blue2[0],self.position_blue2[1], 0.1, 0)
							# com.send_data(send_interact_blue2)

						# 步兵4号
						elif self.output[2] == 3:
							self.position_blue3 = point2d
							self.track_identities_blue3 = 104
							# send_interact_blue3 = com_class.lmap_interaction_send(self.track_identities_blue3,self.position_blue3[0],self.position_blue3[1], 0.1, 0)
							# com.send_data(send_interact_blue3)

						# 步兵5号
						# elif self.output[2] == 4:
						# 	self.position_blue4 = point2d
						# 	self.track_identities_blue4 = 105
							# send_interact_blue4 = com_class.lmap_interaction_send(self.track_identities_blue4,self.position_blue4[0], self.position_blue4[1], 0.1, 0)
							# com.send_data(send_interact_blue4)

						# 哨兵7号
						# elif self.output[2] == 5:
						# 	self.position_blue5 = point2d
						# 	self.track_identities_blue5 = 107
							# send_interact_blue5 = com_class.lmap_interaction_send(self.track_identities_blue5,self.position_blue5[0], self.position_blue5[1], 0.1, 0)
							# com.send_data(send_interact_blue5)

					if (len(self.output) > 0) and (self.output[2] in self.send_list_red) and self.T3:
						# 红方
						point2d = self.output[0:2]  # 位置
						# 李树程3
						point2d[0] = self.x_length - (point2d[0] - self.lidar_pos_x)
						point2d[1] = self.y_length - point2d[1] - self.lidar_pos_y
						# 英雄1号
						if self.output[2] == 6:
							self.position_red0 = point2d
							self.track_identities_red0 = 1
							# send_interact_red0 = com_class.lmap_interaction_send(self.track_identities_red0,self.position_red0[0],self.position_red0[1], 0.1, 0)
							# com.send_data(send_interact_red0)

						# 工程2号
						# elif self.output[2] == 7:
						# 	self.position_red1 = point2d
						# 	self.track_identities_red1 = 2
							# send_interact_red1 = com_class.lmap_interaction_send(self.track_identities_red1,self.position_red1[0],self.position_red1[1], 0.1, 0)
							# com.send_data(send_interact_red1)

						# 步兵3号
						elif self.output[2] == 8:
							self.position_red2 = point2d
							self.track_identities_red2 = 3
							# send_interact_red2 = com_class.lmap_interaction_send(self.track_identities_red2,self.position_red2[0],self.position_red2[1], 0.1, 0)
							# com.send_data(send_interact_red2)

						# 步兵4号
						elif self.output[2] == 9:
							self.position_red3 = point2d
							self.track_identities_red3 = 4
							# send_interact_red3 = com_class.lmap_interaction_send(self.track_identities_red3,self.position_red3[0],self.position_red3[1], 0.1, 0)
							# com.send_data(send_interact_red3)

						# 步兵5号
						# elif self.output[2] == 10:
						# 	self.position_red4 = point2d
						# 	self.track_identities_red4 = 5
							# send_interact_red4 = com_class.lmap_interaction_send(self.track_identities_red4,self.position_red4[0],self.position_red4[1], 0.1, 0)
							# com.send_data(send_interact_red4)

						# 哨兵7号
						# elif self.output[2] == 11:
						# 	self.position_red5 = point2d
						# 	self.track_identities_red5 = 7
							# send_interact_red5 = com_class.lmap_interaction_send(self.track_identities_red5,self.position_red5[0],self.position_red5[1], 0.1, 0)
							# com.send_data(send_interact_red5)



					self.T3 = False
					#t1 = time.time()
					#print("耗时为：",t1 - t0)

			#Stream results可视化
			im0=annotator_first.result()
			cv2.namedWindow("Detect",0)
			cv2.imshow("Detect",im0)
			cv2.waitKey(1)


####################################！！！！！！以上为检测部分！！！！！！##########################################################################################



class lidar_node:
	def __init__(self):
		self.detector=yolo_myself()
		##################！！！！！！！通信！！！！！！#######################################

		# global com
		# com = Port('/dev/ttyUSB0',115200)
		# com.openPort()
		# global com_class
		# com_class = Communicator()

		self.P = [[1922.3, 0, 1580],
		 [0, 1923, 1023.6],
		 [0, 0, 1]]
		self.K = [-0.4551, 0.2824, -0.0017, 0.0026, -0.1042]
	
	def yolo_detect_callback(self,data):
		t0 = time.time()
		image_np = np.frombuffer(data.data, dtype=np.uint8)
		frame = image_np.reshape((data.height, data.width, 3))
		# frame = cv2.undistort(frame, np.array(self.P), np.array(self.K))
		#调用yolo检测
		#t1=time.time()
		# print("处理图像耗时：", t1- t0)
		self.detector.yolt_callback(frame)
		t2 = time.time()
		print("图像处理耗时：",t2-t0)



def main():
	try:
		rospy.init_node("NUDT_lidar", anonymous=True)
		lidar = lidar_node()
		rospy.Subscriber("/dense_point_cloud_topic", PointCloud2, lidar.detector.pointcloud_callback)
		rospy.Subscriber("image_topic", Image, lidar.yolo_detect_callback)
		rospy.spin()
	finally:
		cv2.destroyAllWindows()


main()

"""
通信机制修改方案，我的设想：
规则修改：改为单包发送六台机器人的坐标，频率限制在3-5hz
1. 哨兵盲区预测
2. 工程根据对手特点发位置（银矿、金矿），兑换点？
3. 英雄加上三个步兵，检测到更新位置
4. 仍采用timer发送
5. 发送的包的内容要修改
"""

"""
0.v0 是完好不动的区域赛比赛代码（代码备份，保底代码）
  v1 是用于修改的代码模板
1.v2是在v1的基础上修改了点云投影的方式，1帧只投影一次

2024-6-30
最终的通信机制修改了
"""
