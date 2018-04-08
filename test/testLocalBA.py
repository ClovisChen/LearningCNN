#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pose.LocalBA
import pose.Frame
import pose.MapPoint
import BA_utils
import sophus
import numpy as np
import pose.PyLie.transform
import time


def test_ba():
    fileName = "../data/problem-49-7776-pre.txt"
    camera_params, points_3d, camera_indices, point_indices, points_2d = BA_utils.read_bal_data(fileName)
    ba = pose.LocalBA.LocalBA()
    for cam_idx, cam in enumerate(camera_params):
        phi = cam[:3]
        theta = np.linalg.norm(phi)
        if theta == 0:
            alpha = np.array([1, 0, 0])
        else:
            alpha = phi / theta
        _pose = pose.PyLie.transform.rotation_matrix(theta, alpha)
        _pose[:3, 3] = cam[3:6]
        _cam = cam[6:]
        frame = pose.Frame.Frame()
        frame.mPose2World = _pose
        frame.cam = _cam
        ba.add_frame(frame)

    for pt_idx, pt in enumerate(points_3d):
        map_pt = pose.MapPoint.MapPoint(pt)
        ba.add_map_point(map_pt)
    # print len(ba.nKFs)
    # print len(ba.mMapPoints)
    ba.n_observations = point_indices.shape[0]
    ba.intrinsics = camera_params[:, :3]
    assert len(camera_indices) == len(point_indices)
    for pt2d_idx, (_cam_idx, _pt_idx) in enumerate(zip(camera_indices, point_indices)):
        feat = pose.MapPoint.Feature(points_2d[pt2d_idx], _pt_idx)
        assert _cam_idx in ba.nKFs.iterkeys()
        ba.nKFs[_cam_idx].add_feature(feat)

    ba.stack_data()
    ba.set_point_indices()
    A = ba.bundle_adjustment_sparsity(ba.camera_indices, ba.point_indices)
    tic = time.clock()
    ba.run_int(A)
    toc = time.clock()
    print "time cost ", toc - tic


def test_ba_matrix():
    fileName = "../data/problem-49-7776-pre.txt"
    camera_params, points_3d, camera_indices, point_indices, points_2d = BA_utils.read_bal_data(fileName)
    ba = pose.LocalBA.LocalBA()
    for cam_idx, cam in enumerate(camera_params):
        phi = cam[:3]
        theta = np.linalg.norm(phi)
        if theta == 0:
            alpha = np.array([1, 0, 0])
        else:
            alpha = phi / theta
        _pose = pose.PyLie.transform.rotation_matrix(theta, alpha)
        _pose[:3, 3] = cam[3:6]
        _cam = cam[6:]
        frame = pose.Frame.Frame()
        frame.mPose2World = _pose
        frame.cam = _cam
        ba.add_frame(frame)

    for pt_idx, pt in enumerate(points_3d):
        map_pt = pose.MapPoint.MapPoint(pt)
        ba.add_map_point(map_pt)

    ba.n_observations = point_indices.shape[0]
    ba.intrinsics = camera_params[:, :3]
    assert len(camera_indices) == len(point_indices)
    for pt2d_idx, (_cam_idx, _pt_idx) in enumerate(zip(camera_indices, point_indices)):
        feat = pose.MapPoint.Feature(points_2d[pt2d_idx], _pt_idx)
        assert _cam_idx in ba.nKFs.iterkeys()
        ba.nKFs[_cam_idx].add_feature(feat)

    # ba.stack_data()
    ba.set_point_indices()
    A = ba.bundle_adjustment_sparsity(ba.camera_indices, ba.point_indices)
    tic = time.clock()
    ba.run(A)
    toc = time.clock()
    print "time cost ", toc - tic


def test_plane(noise, use_ransac=False):
    point_num = 100
    ptx = np.random.random([point_num, 1]) * 10
    pty = np.random.random([point_num, 1]) * 10
    ptz = 0.1 * (1 - 5 * ptx - 10 * pty) + np.random.random([point_num, 1]) * noise
    ptxy = np.append(ptx, pty, axis=1)
    ptxyz = np.append(ptxy, ptz, axis=1)
    ba = pose.LocalBA.LocalBA()
    if use_ransac:
        normal = ba.fit_plane(ptxyz)
    else:
        normal = ba.fit_plane_ransac(30, 5, 0.05, 40, ptxyz, debug=False)

    norm_val = np.linalg.norm(normal[:3])
    normal /= norm_val

    f = 300
    c = 300
    camera = np.array([[f, 0, c], [0, f, c], [0, 0, 1]])
    pose2World = sophus.SE3.rotX(0 * np.pi / 4) * sophus.SE3.trans(0.1, 0.3, 0.5)
    pose2Cam = pose2World.inverse()
    PosImageArray = np.empty([point_num, 2])
    for pt_idx, pt in enumerate(ptxyz):
        pt = np.append(pt, 1)
        cam_pos = np.dot(pose2Cam.matrix(), pt)
        PosImageArray[pt_idx, :] = cam_pos[:2] / cam_pos[2] * f + c
    error, H = ba.homograph_error(ptxyz, PosImageArray, pose2Cam.matrix(), camera)
    print ptxyz[0, :]
    print PosImageArray[0, :]
    # print pose2Cam.matrix(), H
    print np.average(error)
    # print error


def test_plane_with_time():
    import time

    tic = time.clock()
    test_plane(0.1, True)
    toc = time.clock()
    print 'time cost', toc - tic
    tic = time.clock()
    test_plane(0.1, False)
    toc = time.clock()
    print 'time cost', toc - tic


if __name__ == '__main__':
    # test_plane_with_time()
    test_ba()
    # test_ba_matrix()
