#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pose.Frame
import data.pose_utils
import numpy as np
import pose.PyLie.Quaternion
import cv2
import pose.track
from test_utils import *
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import std_msgs.msg

cloud_topic = '/mapping/cloud'
pose_topic = '/mapping/pose'
path_topic = '/mapping/path'
path_topic_est = '/mapping/path_est'
pose_topic_est = '/mapping/pose_est'


def test_points():
    data_root = "/home/bobin/data/kitti"
    for sequence in range(4, 5):
        trajectory_filename = '%s/sequences/%.2d/%.2d.txt' % (data_root, sequence, sequence)
        timeStamp_filename = '%s/sequences/%.2d/times.txt' % (data_root, sequence)
        timeVec = data.pose_utils.load_timeStamp(timeStamp_filename)

        trajectory = data.pose_utils.load_trajectory_mat(trajectory_filename)
        frame_list = list()
        for line, frame_id in enumerate(trajectory):
            image_file_left = '%s/sequences/%.2d/image_0/%.6d.png' % (data_root, sequence, frame_id)
            image_file_right = '%s/sequences/%.2d/image_1/%.6d.png' % (data_root, sequence, frame_id)
            left_image = cv2.imread(image_file_left, cv2.IMREAD_GRAYSCALE)
            right_image = cv2.imread(image_file_right, cv2.IMREAD_GRAYSCALE)
            depth_image = np.zeros(left_image.shape)
            frame = pose.Frame.Frame(timeVec[frame_id], left_image, right_image, depth_image)
            # frame.left_pyr()
            # frame.depth_Pyr()
            frame.point_select_grid()
            cv2.imshow("test", frame.mLeftImage)
            cv2.imshow("test2", frame.mark_pyr_points(0))

            cv2.waitKey(10)
            gtPose = np.eye(4)
            gtPose[:3, :] = np.reshape(line, (3, 4))
            frame.set_gt_pose(gtPose)
            frame_list.append(frame)


def test_direct():
    pose_pub = rospy.Publisher(pose_topic, PoseStamped, queue_size=1)
    pose_pub_est = rospy.Publisher(pose_topic_est, PoseStamped, queue_size=1)
    path_pub = rospy.Publisher(path_topic, Path, queue_size=1)
    path_pub_est = rospy.Publisher(path_topic_est, Path, queue_size=1)
    rospy.init_node('mapping', anonymous=True)
    rospy.loginfo("Start Mapping")

    data_root = "/home/bobin/data/rgbd/tum/rgbd_dataset_freiburg1_xyz/"
    trajectory_filename = data_root + 'associate-rgb-tra.txt'
    image_path = data_root + 'associate-rgb-depth.txt'
    image_files = load_image_path(image_path)
    trajectory = load_trajectory(trajectory_filename)
    keys = None
    d = [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
    camera = pose.Frame.Camera(517.3, 516.5, 318.6, 255.3, 640, 480, d, None)
    pyr_camera = pose.Frame.PyrCamera(camera, 4)
    if len(trajectory) < len(image_files):
        keys = sorted(trajectory.iterkeys())
    else:
        keys = sorted(image_files.iterkeys())
    tracker = pose.track.Tracker(camera)
    path_msg = Path()
    path_msg_est = Path()
    for frame_id, key in enumerate(keys):
        if not image_files.has_key(key):
            continue
        image = cv2.imread(data_root + image_files[key][0], cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(data_root + image_files[key][1], cv2.IMREAD_UNCHANGED)
        if not trajectory.has_key(key):
            continue
        gtPose = trajectory[key]
        frame = pose.Frame.Frame(key, image, None, depth, pyr_camera)
        qx, qy, qz, qw = gtPose[3:]
        rot = pose.PyLie.Quaternion.quat2mat([qw, qx, qy, qz])
        gt_pose_mat = np.eye(4)
        gt_pose_mat[:3, :3] = rot
        gt_pose_mat[:3, 3] = gtPose[:3]
        frame.set_gt_pose(gt_pose_mat)
        frame.point_select_grid()
        pose_est = tracker.insert_frame(frame, frame_id)
        # H = tracker.LKTrack(frame)
        # tracker.PoseGaussianNewton()
        # 如何将当前帧添加到 frame array？
        # tracker.FrameArray[frame_id] = frame
        pose_msg = PoseStamped()
        pose_msg.pose.position.x = gtPose[0]
        pose_msg.pose.position.y = gtPose[1]
        pose_msg.pose.position.z = gtPose[2]
        w, x, y, z = data.pose_utils.rot2quat(rot)
        pose_msg.pose.orientation.w = w
        pose_msg.pose.orientation.x = x
        pose_msg.pose.orientation.y = y
        pose_msg.pose.orientation.z = z
        pose_msg.header = std_msgs.msg.Header()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = 'mapping'
        pose_pub.publish(pose_msg)

        pose_msg_est = PoseStamped()
        pose_msg_est.pose.position.x = pose_est[0, 3]
        pose_msg_est.pose.position.y = pose_est[1, 3]
        pose_msg_est.pose.position.z = pose_est[2, 3]
        w, x, y, z = data.pose_utils.rot2quat(pose_est[:3, :3])
        pose_msg_est.pose.orientation.w = w
        pose_msg_est.pose.orientation.x = x
        pose_msg_est.pose.orientation.y = y
        pose_msg_est.pose.orientation.z = z
        pose_msg_est.header = std_msgs.msg.Header()
        pose_msg_est.header.stamp = rospy.Time.now()
        pose_msg_est.header.frame_id = 'mapping'
        pose_pub_est.publish(pose_msg_est)

        path_msg.header = pose_msg.header
        path_msg.poses.append(pose_msg)
        path_pub.publish(path_msg)

        path_msg_est.header = pose_msg.header
        path_msg_est.poses.append(pose_msg_est)
        path_pub_est.publish(path_msg_est)

        rospy.sleep(0.1)


if __name__ == '__main__':
    test_direct()
    # test_points()
