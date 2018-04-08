#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.optimize import least_squares
import numpy as np


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


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


class PointOptimize():
    def __init__(self):
        self.points = None
        self.poses = None
        self.cam = None

    @staticmethod
    def project(homograph, ptw, pt2d):
        assert len(pt2d) == len(ptw)
        homo_matrix = homograph.reshape((-1, 3))
        error = pt2d - homo_matrix.dot(ptw)
        return error.ravel()

    def homo_optimize(self, homo, plane_pt, image_pt):
        res = least_squares(self.project, homo, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            loss='huber', args=(plane_pt, image_pt))
        return res.x


class PoseOptimize():
    def __init__(self, _pose, _cam):
        self.pose = _pose
        self.cam = _cam

    @staticmethod
    def proj(data, pt3d, pt2d):
        camera = np.zeros((1, 9))
        camera[0][:6] = data
        camera[0][6] = 1
        points_proj = project(pt3d, camera)
        return (pt2d - points_proj).ravel()

    # def project(self, data, pt3d, pt2d):
    #     assert len(pt3d) == len(pt2d)
    #     n_obs = len(pt2d)
    #     rot = data[:3]
    #     angle = np.linalg.norm(rot)
    #     if angle != 0:
    #         direction = rot / angle
    #     _pose = tf.rotation_matrix(angle, direction)
    #     _pose[:3, 3] = data[3:]
    #     one = np.ones((1, n_obs))
    #     pt3dh = np.concatenate((pt3d.T, one), axis=0)
    #     proj = _pose.dot(pt3dh)[:3, :]
    #     proj /= proj[2, :]
    #     pt = self.cam.dot(proj)
    #     error = pt2d - pt[:2, :].T
    #     return error.ravel()

    def optimize(self, pt3d, pt2d):
        res = least_squares(self.proj, self.pose, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            loss='huber', args=(pt3d, pt2d))
        print res.optimality
        if res.optimality < 1e-3:
            self.pose = res.x

    def jac(self, pt3d):
        n_obs = len(pt3d)
        one = np.ones((1, n_obs))
        pt3dh = np.concatenate((pt3d.T, one), axis=0)
        proj = self.pose.dot(pt3dh)[:3, :]
        fx = self.cam[0, 0]
        fy = self.cam[1, 1]

        jac_mat = np.zeros((2 * n_obs, 6))
        i = np.arange(n_obs)
        x = proj[:, 0]
        y = proj[:, 1]
        z_inv = 1.0 / proj[:, 2]
        zz_inv = z_inv ** 2
        jac_mat[i * 2, 0] = -fx * x * y * zz_inv
        jac_mat[i * 2, 1] = fx * (1 + x * x * zz_inv)
        jac_mat[i * 2, 2] = -fx * y * z_inv
        jac_mat[i * 2, 3] = fx * z_inv
        jac_mat[i * 2, 5] = - fx * x * zz_inv

        jac_mat[i * 2 + 1, 0] = -fy(1 + y * y * zz_inv)
        jac_mat[i * 2 + 1, 1] = fy * x * y * zz_inv
        jac_mat[i * 2 + 1, 2] = fx * x * z_inv
        jac_mat[i * 2 + 1, 4] = fy * z_inv
        jac_mat[i * 2 + 1, 5] = - fy * y * zz_inv
        return jac_mat
