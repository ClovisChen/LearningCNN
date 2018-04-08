#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import std_msgs.msg
import matplotlib.pyplot as plt
import cv2
import time
import depth_mapping
import net.utils.utils
import data.pose_utils

scan_path = '/media/bobin/Seagate/data/slam/kitti_raw/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/'

cloud_topic = '/mapping/cloud'
pose_topic = '/mapping/pose'

path_topic = '/mapping/path'
scan_num = 108


def test_kitti_velo():
    cloud_pub = rospy.Publisher(cloud_topic, PointCloud)
    rospy.init_node('mapping', anonymous=True)
    rospy.loginfo("Start Mapping")
    while not rospy.is_shutdown():
        for i in range(scan_num):
            velo_file = scan_path + '%.10d.bin' % i
            scan = np.fromfile(velo_file, dtype=np.float32)
            # while not rospy.is_shutdown():
            cloud = PointCloud()
            cloud.header = std_msgs.msg.Header()
            cloud.header.stamp = rospy.Time.now()
            cloud.header.frame_id = "mapping"
            scan = scan.reshape((-1, 4))
            point_num = len(scan)
            cloud.points = [None] * point_num
            for i in range(point_num):
                x, y, z, c = scan[i]
                cloud.points[i] = Point(x, y, z)

            cloud_pub.publish(cloud)
            rospy.sleep(0.1)


def test_depth_mapping():
    pose_pub = rospy.Publisher(pose_topic, PoseStamped)
    path_pub = rospy.Publisher(path_topic, Path)
    rospy.init_node('mapping', anonymous=True)
    rospy.loginfo("Start Mapping")

    mapping = depth_mapping.Mapping()
    params = net.monodepth_main.monodepth_parameters(
        encoder='vgg',
        height=256,
        width=512,
        batch_size=8,
        num_threads=8,
        num_epochs=50,
        do_stereo=False,
        wrap_mode='border',
        use_deconv=False,
        alpha_image_loss=0.85,
        disp_gradient_loss_weight=0.1,
        lr_loss_weight=1.0,
        full_summary=True)
    root_path = '/home/chen-tian/data/data/code/learningReloc/'
    data_path = '/home/chen-tian/data/data/KITTI/odom/'
    test_params = net.utils.utils.test_parameters(
        root_path=root_path,
        data_path=data_path,
        filenames_file=root_path + 'net/utils/filenames//kitti_odom_color_depth.txt',
        dataset='kitti',
        mode='test',
        checkpoint_path=root_path + 'net/data/model/model_kitti',
        log_directory=root_path + 'learningReloc/log/',
        output_directory=root_path + 'learningReloc/output/',
        kitti_calib=data_path + 'dataset/sequences/00/calib.txt',
        trajectory_file=data_path + 'dataset/poses/00.txt',
        height_origin=370,
        width_origin=1226,
        calib_ext_file='',
        calib_int_file='',
        ground_truth_image=''
    )

    num_test_samples = mapping.build_net(params, test_params)
    mapping.load_trajectory(test_params.trajectory_file)
    mapping.calib_params(test_params.kitti_calib)
    assert num_test_samples == len(mapping.trajectory)
    print('now testing {} files'.format(num_test_samples))
    # disparities_vector = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    # disparities_pp_vector = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    path_msg = Path()

    for step in range(num_test_samples):
        disp, left_image = mapping.sess_run()
        disparities = disp[0].squeeze()
        plt.imshow(disparities)
        plt.pause(0.001)
        left_ori = np.uint8(left_image[0] * 255)
        depth = mapping.bf / disparities
        cv2.imshow('left', left_ori)
        cv2.waitKey(10)
        color_origin = np.uint8(left_image[0] * 255)
        gray_orgin = cv2.cvtColor(color_origin, cv2.COLOR_BGR2GRAY)
        pose_mat = np.eye(4)
        pose_mat[:3, :] = mapping.trajectory[step].reshape((3, 4))
        pcd = net.utils.utils.triangulate(gray_orgin, depth, pose_mat, mapping.intrinsic,
                                          (test_params.width_origin, test_params.height_origin))
        cloud = PointCloud()
        cloud.header = std_msgs.msg.Header()
        cloud.header.stamp = rospy.Time.now()
        cloud.header.frame_id = "mapping"
        point_num = len(pcd)
        cloud.points = [None] * point_num
        for i in range(point_num):
            x, y, z, w = pcd[i]
            cloud.points[i] = Point(x, y, z)
        cloud_pub.publish(cloud)

        pose_msg = PoseStamped()
        pose_msg.pose.position.x = pose_mat[0, 3]
        pose_msg.pose.position.y = pose_mat[1, 3]
        pose_msg.pose.position.z = pose_mat[2, 3]
        w, x, y, z = data.pose_utils.rot2quat(pose_mat[:3, :3])
        pose_msg.pose.orientation.w = w
        pose_msg.pose.orientation.x = x
        pose_msg.pose.orientation.y = y
        pose_msg.pose.orientation.z = z
        pose_msg.header = std_msgs.msg.Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'mapping'
        pose_pub.publish(pose_msg)
        path_msg.header = pose_msg.header
        path_msg.poses.append(pose_msg)
        path_pub.publish(path_msg)
        rospy.sleep(0.001)
        # rospy.spin()


if __name__ == '__main__':
    test_depth_mapping()
