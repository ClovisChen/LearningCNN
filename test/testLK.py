#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pose.Frame
import numpy as np
import cv2
import pose.track
import data.pose_utils
from test_utils import *


def test_direct():
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

    for frame_id, key in enumerate(keys):
        if not image_files.has_key(key):
            continue
        image = cv2.imread(data_root + image_files[key][0], cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(data_root + image_files[key][1], cv2.IMREAD_UNCHANGED)
        if not trajectory.has_key(key):
            continue
        gt_pose = trajectory[key]
        frame = pose.Frame.Frame(key, image, None, depth, pyr_camera)
        qx, qy, qz, qw = gt_pose[3:]
        rot = data.pose_utils.quat2mat([qw, qx, qy, qz])
        gt_pose_mat = np.eye(4)
        gt_pose_mat[:3, :3] = rot
        gt_pose_mat[:3, 3] = gt_pose[:3]
        frame.set_gt_pose(gt_pose_mat)
        mark_image = tracker.lk_track(frame)
        if mark_image is None:
            continue
        cv2.imshow('lk track', mark_image)
        cv2.waitKey(0)


if __name__ == '__main__':
    test_direct()
    # test_points()
