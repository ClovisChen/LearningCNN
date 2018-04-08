#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import EurocReader as reader
import pose.IMUPreInt as imu
import rospy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg._Image import Image
from nav_msgs.msg import Path
import std_msgs.msg
import cv2

pose_topic = '/IMUTest/pose'
path_topic = '/IMUTest/path'
path_topic_est = '/IMUTest/path_est'
pose_topic_est = '/IMUTest/pose_est'
image_topic = 'IMUTest/image'


def test_camera_imu():
    pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
    pose_pub_est = rospy.Publisher(pose_topic_est, PoseStamped, queue_size=1)
    path_pub = rospy.Publisher(path_topic, Path, queue_size=1)
    path_pub_est = rospy.Publisher(path_topic_est, Path, queue_size=1)
    image_pub = rospy.Publisher(image_topic, Image, queue_size=1)
    rospy.init_node('IMUTest', anonymous=True)
    rospy.loginfo("Start IMUTest")

    euroc_root_dir = '/home/bobin/data/euroc/MH_01/'
    euroc_imu_dir = euroc_root_dir + 'imu0/data.csv'
    euroc_gt_dir = euroc_root_dir + 'state_groundtruth_estimate0/data.csv'
    euroc_img_list_dir = euroc_root_dir + 'cam0/data.csv'
    imu_origin_data = reader.read_imu_data(euroc_imu_dir)
    pose_gt_data = reader.read_ground_truth(euroc_gt_dir)
    ## len(image_list) == len(rot_imu_int)
    ## len(image_list) == len(pos_imu_int)
    ## len(image_list) == len(speed_imu_int)
    ## len(image_list) == len(bias_acc)
    ## len(image_list) == len(bias_gyro)
    image_list = reader.read_images_lists(euroc_img_list_dir)
    gt_index = 0
    imu_index = 0
    imu_preint = None
    num_poses = len(image_list)
    rot_imu_int = np.empty((num_poses, 3, 3))
    pos_imu_int = np.empty((num_poses, 3))
    speed_imu_int = np.empty((num_poses, 3))
    rot_imu_int[0] = np.eye(3)

    bias_acc = np.zeros((num_poses, 3))
    bias_gyro = np.zeros((num_poses, 3))
    gravity = np.zeros((num_poses, 3))

    ## we need a initialized results of acc, gyro bias and gravity.

    for index, (stamp, image_name) in enumerate(image_list):
        imu_int_index = 0
        stamp = float(stamp)
        ## update acc bias and gyro bias
        bias_acc_current = bias_acc[index]
        bias_gyro_current = bias_gyro[index]
        while pose_gt_data[gt_index][0] < stamp:
            gt_index += 1
            # while imu_origin_data[imu_index][0] < stamp:
            #     imu_int_index += 1
            #     imu_index += 1
            # ## pre-intergration of imu
            # ## fix acc and gyro bias
            # imu_item = imu_origin_data[imu_index]
            # _gyro = imu_item[1:4]
            # _acc = imu_item[4:]
            # _time = imu_item[0]
            # imu_data_item = imu.IMUData(_gyro=_gyro-bias_gyro_current, _acc=_acc-bias_acc_current, _time_stamp=_time)
            # if imu_int_index == 0:
            #     imu_preint = imu.IMUPreInt(imu_data_item)
            # else:
            #     imu_preint = imu_preint + imu_data_item
        # if index > 0:
        #     rot_current = rot_imu_int[index-1].dot(imu_preint.delta_rot)
            # speed_current = speed_imu_int[index-1] +
            # rot_imu_int[index] = rot_current
            # pos_imu_int[index] = pos_imu_int[index-1] + rot_current *
        img = cv2.imread(euroc_root_dir + 'cam0/data/' + image_name)
        cv2.imshow('test', img)
        cv2.waitKey(10)
        gt_item = pose_gt_data[gt_index]
        _quat = gt_item[4:8]
        _pos = gt_item[1:4]
        _time = gt_item[0]
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = _pos[0]
        pose_msg.pose.position.y = _pos[1]
        pose_msg.pose.position.z = _pos[2]

        pose_msg.pose.orientation.w = _quat[0]
        pose_msg.pose.orientation.x = _quat[1]
        pose_msg.pose.orientation.y = _quat[2]
        pose_msg.pose.orientation.z = _quat[3]
        pose_msg.header = std_msgs.msg.Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'IMUTest'
        pose_pub.publish(pose_msg)


        rospy.sleep(0.001)


if __name__ == '__main__':
    test_camera_imu()
