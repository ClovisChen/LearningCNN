#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import SE3_XYZ as se3_xyz
import pose.PyLie.transform as tf
import pose.PyLie.SE3 as SE3


def test_optimize_pose(point_num):
    ptx = np.linspace(0, 100, point_num)
    pty = np.random.uniform(0, 10, point_num)
    ptz = 1 - 0.5 * ptx - 0.5 * pty + np.random.random(point_num)
    ptxy = np.concatenate((ptx[:, np.newaxis], pty[:, np.newaxis]), axis=1)
    pt_xyz_w = np.concatenate((ptxy, ptz[:, np.newaxis]), axis=1)

    pose_xyz_rot = np.array([[0, np.pi / 6.0, 0, 0, 0, 10]])
    focal = 1
    camera = np.zeros((1, 9))
    camera[0][:6] = pose_xyz_rot
    camera[0][6] = 1
    pt_xyz_c = se3_xyz.rotate(pt_xyz_w, pose_xyz_rot[np.newaxis, 0, :3]) + \
               pose_xyz_rot[np.zeros(point_num, dtype=int), 3:]
    pt_uv = focal * pt_xyz_c[:, :2] / pt_xyz_c[:, 2, np.newaxis]
    # pt_uv = ba.project(pt_xyz_w, camera)
    _pose = np.array([0, np.pi / 6.2, 0.1, 0.1, 0, 10.4])
    _cam = np.eye(3)
    motion_ba = se3_xyz.PoseOptimize(_pose, _cam)
    motion_ba.optimize(pt_xyz_w, pt_uv)
    rlt = motion_ba.pose
    print rlt
    print np.pi / 6


def test_homo_ba(point_num):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    point = np.array([1, 2, 3])
    normal = np.array([1, 1, 2])

    point2 = np.array([10, 50, 50])

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2] + np.random.random(10)

    vec = np.array([1, 0., 0.])
    vec_noise = vec + np.array([0.1, 0.2, 0.3])
    theta = 0.1
    trans = np.array([0., 10, 0])
    trans_noise = trans + np.array([1, 2, 3])
    R = tf.rotation_matrix(theta, vec)[:3, :3]
    R_noise = tf.rotation_matrix(theta, vec_noise)[:3, :3]
    H = R + normal.dot(SE3.skew_matrix(trans)) / d
    H_noise = R_noise + normal.dot(SE3.skew_matrix(trans_noise)) / d
    plane_pt = np.array([xx, yy, z]).reshape((3, -1))

    im_pt = H.dot(plane_pt)
    struct_ba = se3_xyz.PointOptimize()
    hest = struct_ba.homo_optimize(H_noise.ravel(), plane_pt, im_pt)
    print hest.reshape((-1, 3))
    print H
    print H_noise


    # plt.figure().gca(projection='3d')
    # ax0 = plt.gca()
    # ax0.hold(True)
    # x0, y0, z0 = im_pt
    # ax0.scatter(x0, y0, z0, color='green')
    # ax0.scatter(xx, yy, z, color='red')
    #
    # # plot the surface
    # plt3d = plt.figure().gca(projection='3d')
    # plt3d.plot_surface(xx, yy, z, alpha=0.2)
    #
    # # and i would like to plot this point :
    # ax = plt.gca()
    # ax.hold(True)
    #
    # # ax.scatter(point2[0], point2[1], point2[2], color='green')
    # plt.show()


if __name__ == '__main__':
    # test_optimize_pose(100)
    test_homo_ba(100)
