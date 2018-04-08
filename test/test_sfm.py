#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from pose.struct import LineStruct
from pose.OpticalFlow import FrameLK, Camera, Pose
from glob import glob
import numpy as np
import time
from pose.track import Tracker
from data.yaml_read import *
import yaml
from data.kitti_reader import KITTIData
import pose.settings as settings
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def test_extract(img):
    tic = time.clock()
    lines = LineStruct.detect_lines(img)
    toc = time.clock()
    print 'cost time', toc - tic
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    LineStruct.draw_lines(color, lines)
    cv2.imshow('lines', color)


def collect_data(frames):
    observations = None
    obs_point_index = None
    obs_frame_index = None
    poses = np.empty((len(frames), 6))
    for cnt, frame in enumerate(frames):
        points_num = len(frame.points)
        poses[cnt, :3] = frame.rot_trans.rot
        poses[cnt, 3:] = frame.rot_trans.trans
        if observations is None:
            observations = frame.points
            obs_point_index = frame.point_index
            obs_frame_index = np.ones(points_num, dtype=int) * cnt

        else:
            observations = np.concatenate((observations, frame.points), axis=0)
            obs_point_index = np.concatenate((obs_point_index, frame.point_index), axis=0)
            obs_frame_index = np.concatenate((obs_frame_index, np.ones(points_num, dtype=int) * cnt), axis=0)
    return poses, observations, obs_frame_index, obs_point_index


def set_frames_poses(poses, frames):
    assert len(poses) == len(frames)
    for pose, frame in zip(poses, frames):
        frame.rot_trans.rot = pose[:3]
        frame.rot_trans.trans = pose[3:]
        frame.pose[:3, :3], _ = cv2.Rodrigues(pose[:3])
        frame.pose[:3, 3] = pose[3:]


def rotate(points, rot_vecs):
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, poses, camera):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, poses[:, :3])
    points_proj += poses[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    cxy = np.array([[camera.cx, camera.cy]])
    fxy = np.array([[camera.fx, camera.fy]])
    points_proj *= fxy
    points_proj += cxy
    return points_proj


def residual(params, n_cameras, n_points, pose_indices, point_indices, points_2d, camera):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    pose_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], pose_params[pose_indices], camera)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, pose_indices, point_indices):
    dof_pose = 6
    dof_point = 3
    m = pose_indices.size * 2
    n = n_cameras * dof_pose + n_points * dof_point

    A = lil_matrix((m, n), dtype=int)
    i = np.arange(pose_indices.size)
    for s in range(dof_pose):
        A[2 * i, pose_indices * dof_pose + s] = 1
        A[2 * i + 1, pose_indices * dof_pose + s] = 1

    for s in range(dof_point):
        A[2 * i, n_cameras * dof_pose + point_indices * dof_point + s] = 1
        A[2 * i + 1, n_cameras * dof_pose + point_indices * dof_point + s] = 1

    return A


def optimise(observations, obs_frame_index, obs_point_index, poses, points, camera):
    n_poses = len(poses)
    n_points = len(points)
    data = np.hstack((poses.ravel(), points.ravel()))
    sparsity = bundle_adjustment_sparsity(n_poses, n_points, obs_frame_index, obs_point_index)
    res = least_squares(residual, data, jac_sparsity=sparsity, verbose=0, x_scale='jac',
                        ftol=1e-4, method='trf',
                        args=(n_poses, n_points, obs_frame_index, obs_point_index, observations, camera))
    params = res.x
    poses = params[:n_poses * 6].reshape((n_poses, 6))
    points = params[n_poses * 6:].reshape((n_points, 3))


if __name__ == '__main__':
    data_root = '/media/bobin/DATA1/SLAM/data/odometry-kitti/'

    traced_pt = None
    last_frame = None
    current_frame = None

    opt_flag = False

    kitti_reader = KITTIData(data_root, sequence=0)
    left_names, right_names = kitti_reader.read_img_list()
    kitti_reader.read_gt_data()
    kitti_reader.read_params()
    camera = Camera(kitti_reader.fx, kitti_reader.fy, kitti_reader.cx, kitti_reader.cy, kitti_reader.bf)
    point_3d = None
    point_3d_num = 0
    frame_queue = list()

    for cnt, (left_name, right_name) in enumerate(zip(left_names, right_names)):
        left = cv2.imread(left_name, cv2.IMREAD_UNCHANGED)
        right = cv2.imread(right_name, cv2.IMREAD_UNCHANGED)
        tic = time.clock()
        current_frame = FrameLK(left, right, camera)
        gt_pose = kitti_reader.gt_data[cnt].reshape((-1, 4))
        current_pose = np.eye(4)
        est_rot = np.zeros(3)
        est_trans = np.zeros(3)
        if last_frame is not None:
            current_frame.trace_last(last_frame)
            if point_3d is not None:
                est_rot, est_trans = current_frame.estimate_current_pose(point_3d, camera)
                print '   pose estimation', est_rot, est_trans
                gt_rot = gt_pose[:3, :3].T
                gt_trans = -gt_rot.dot(gt_pose[:3, 3])
                rot, jac = cv2.Rodrigues(gt_rot)
                print 'gt pose estimation', rot.ravel(), gt_trans
                rot_cw, jac = cv2.Rodrigues(est_rot)
                trans_cw = est_trans
                current_pose[:3, :3] = rot_cw.T
                current_pose[:3, 3] = -rot_cw.T.dot(trans_cw)

        # current_frame.key_frame = True
        current_frame.pose = current_pose
        current_frame.rot_trans = Pose(est_rot, est_trans)
        mask = current_frame.assign_features_grid()
        new_pts = current_frame.detect_gftt_feature(mask)
        lines = LineStruct.detect_lines(left)
        new_pts_3d = current_frame.left_right_trace(new_pts, point_3d_num)
        if point_3d is None:
            point_3d = new_pts_3d
            point_3d_num = len(new_pts_3d)
        else:
            point_3d = np.concatenate((point_3d, new_pts_3d), axis=0)
            point_3d_num += len(new_pts_3d)
        # color = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        # current_frame.draw_points(color)
        # cv2.imshow('origin', color)
        #
        # if mask is not None:
        #     cv2.imshow('mask', mask)
        # cv2.waitKey(10)

        last_frame = current_frame

        toc = time.clock()
        print 'cost time', toc - tic

        if opt_flag is False:
            continue
        if current_frame.key_frame is True:
            frame_queue.append(current_frame)

        if len(frame_queue) > settings.numFrameWindow:
            frame_queue.pop(0)
            poses, observations, obs_frame_index, obs_point_index = collect_data(frame_queue)
            optimise(observations, obs_frame_index, obs_point_index, poses, point_3d, camera)
            set_frames_poses(poses, frame_queue)