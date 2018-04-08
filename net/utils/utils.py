#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import namedtuple
import numpy as np
import cv2
import contextlib

test_parameters = namedtuple('parameters',
                             'root_path, '
                             'data_path, '
                             'filenames_file, '
                             'dataset, '
                             'mode, '
                             'output_directory, '
                             'checkpoint_path, '
                             'log_directory, '
                             'calib_int_file, '
                             'calib_ext_file, '
                             'kitti_calib, '
                             'trajectory_file, '
                             'ground_truth_image, '
                             'height_origin, '
                             'width_origin'
                             )


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def triangulate(_gray, _depth, pose_mat, K, (w_o, h_o)):
    h, w = _gray.shape
    depth = cv2.resize(_depth, (w, h))
    nx, ny = (w, h)
    x = range(nx)
    y = range(ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv, axis=2)
    yv = np.expand_dims(yv, axis=2)
    depth = np.expand_dims(depth, axis=2)
    point_cloud = np.concatenate((xv, yv, depth), axis=2)
    point_cloud = point_cloud.reshape((-1, 3))
    u, v = K[:2, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    pt_num = point_cloud.shape[0]
    uv = np.repeat([[u, v, 0]], pt_num, axis=0)
    focal_length = 0.5 * (fx + fy)
    fxy = np.repeat([[focal_length] * 3], pt_num, axis=0)
    point_cloud -= uv
    point_cloud /= fxy
    point_cloud[:, 0] *= point_cloud[:, 2] * w_o / w
    point_cloud[:, 1] *= point_cloud[:, 2] * h_o / h
    one = np.ones((pt_num, 1))
    point_cloud = np.concatenate((point_cloud, one), axis=1)
    point_cloud = point_cloud.dot(pose_mat.T)

    return point_cloud
