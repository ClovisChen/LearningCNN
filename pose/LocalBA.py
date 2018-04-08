#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.optimize import least_squares
import numpy as np
from scipy.sparse import lil_matrix
import cv2
import sophus
import data.pose_utils
import pose.PyLie.transform


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
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


class LocalBA:
    def __init__(self):
        self.nKFs = dict()
        self.mMapPoints = dict()
        # self.points_2d = np.empty([0, 2])
        self.point_dicts = dict()
        # self.HomographArray = np.empty([0, 9])
        self.data = None
        self.intrinsics = None
        self.points_2d = None
        self.n_observations = 0

    def add_frame(self, frame):
        key = len(self.nKFs)
        self.nKFs[key] = frame

    def add_map_point(self, map_point):
        key = len(self.mMapPoints)
        self.mMapPoints[key] = map_point

    def run(self, A):
        self.stack_data()
        self.point_camera_indices()
        res = least_squares(self.project_rot_matrix, self.data, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4,
                            method='trf')
        return res

    def run_int(self, A):
        res = least_squares(self.project_axis_angle, self.data, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4,
                            method='trf')
        return res

    def point_camera_indices(self):
        """构造一个camera-point对应的indices，用于计算对应的residuals。
            set:
            pt2d = [u, v]
            camera pose is xi
            point world pos is p
            projection model is \pi

            [u, v]^T = pi(xi, p)
            我们把每个这样的方程成为一个observation，每个obs，对应一个点和camera的组合
            camera indices是按照obs的顺序，对应的point要投影的camera的 index
            point indices是当前的obs产生对应的point index
        :return:
        """
        for i in self.nKFs.iterkeys():
            self.point_dicts[i] = [self.nKFs[i].mFeatures[key].mMapPointKey for key in self.nKFs[i].mFeatures]

    def stack_data(self):
        """
        构造了indices，还需要一个空间用于保存对应的点和camera，camera的数量介绍，在10左右，点的数量在1k左右。
        :return:
        """
        mPointArray = np.empty([0, 3])
        mCameraArray = np.empty([0, 9])
        for frame in self.nKFs.itervalues():
            angle, direction, point = pose.PyLie.transform.rotation_from_matrix(frame.mPose2World)
            rot_vec = angle * direction
            trans_vec = frame.mPose2World[:3, 3]
            pose_vec = np.zeros((1, 9))
            pose_vec[0, :3] = rot_vec
            pose_vec[0, 3:6] = trans_vec
            pose_vec[0, 6:] = frame.cam
            mCameraArray = np.append(mCameraArray, pose_vec, axis=0)

        for pt in self.mMapPoints.itervalues():
            point_pos = np.expand_dims(pt.mWorldPos, axis=0)
            mPointArray = np.append(mPointArray, point_pos, axis=0)

        self.data = np.hstack((mCameraArray.ravel(), mPointArray.ravel()))

    def set_point_indices(self):
        """
        这里有些问题，主要是图像点和地图点的对应，应该如何做？
        :return:
        """
        self.points_2d = np.empty((self.n_observations, 2), dtype=float)
        self.point_indices = np.empty(self.n_observations, dtype=int)
        self.camera_indices = np.empty(self.n_observations, dtype=int)
        count = 0
        for i in self.nKFs.iterkeys():
            for fkey in self.nKFs[i].mFeatures.iterkeys():
                feat = self.nKFs[i].mFeatures[fkey]
                self.point_indices[count] = feat.mMapPointKey
                self.camera_indices[count] = i
                self.points_2d[count, :] = feat.mPixel
                count += 1

    def jac_ba(self, params):
        n_cameras = len(self.nKFs)
        n_points = len(self.mMapPoints)
        m = self.camera_indices.size * 2
        n = n_cameras * 9 + n_points * 3

        jac = np.zeros((m, n))
        i = np.arange(self.camera_indices.size)
        fx = self.intrinsics[:, 0]
        fy = fx
        jac_uv = np.zeros((2, 3))
        jac_uv[0, 0] = fx

        jac[2 * i, self.camera_indices * 9 + 3] = 1
        jac[2 * i, self.camera_indices * 9 + 4] = 0
        jac[2 * i, self.camera_indices * 9 + 5] = 1
        jac[2 * i, self.camera_indices * 9 + 6] = 1
        jac[2 * i, self.camera_indices * 9 + 7] = 1
        jac[2 * i, self.camera_indices * 9 + 8] = 1

        jac[2 * i + 1, self.camera_indices * 9 + 3] = 1
        jac[2 * i + 1, self.camera_indices * 9 + 4] = 1
        jac[2 * i + 1, self.camera_indices * 9 + 5] = 1
        jac[2 * i + 1, self.camera_indices * 9 + 6] = 1
        jac[2 * i + 1, self.camera_indices * 9 + 7] = 1
        jac[2 * i + 1, self.camera_indices * 9 + 8] = 1

        pt_idx = n_cameras * 9 + self.point_indices * 3
        jac[2 * i: 2 * i + 1, pt_idx: pt_idx + 3] = 1
        return jac

    @staticmethod
    def get_point_2d(frame):
        pt2d = np.empty([0, 2])
        for feat in frame.mFeatures.itervalues():
            pixel = np.expand_dims(feat.mPixel, axis=0)
            pt2d = np.append(pt2d, pixel, axis=0)
        return pt2d

    def bundle_adjustment_sparsity(self, camera_indices, point_indices):
        n_cameras = len(self.nKFs)
        n_points = len(self.mMapPoints)
        m = camera_indices.size * 2
        n = n_cameras * 9 + n_points * 3

        A = lil_matrix((m, n), dtype=int)
        i = np.arange(camera_indices.size)
        for s in range(9):
            A[2 * i, camera_indices * 9 + s] = 1
            A[2 * i + 1, camera_indices * 9 + s] = 1

        for s in range(3):
            A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
            A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

        return A

    def project_axis_angle(self, params):
        """
        1. 使用旋转轴+旋转角的方式更加适合批量运算,因为可以进行矩阵,而不必使用循环实现. 这种方式是相当于循环展开, 可以考虑以后并行化.
        2. 由于使用np, 在计算矩阵的cos和sin更快一些. 但是似乎多了很多重复计算. 但是还是要比循环的方法, 似乎是由于自动求导比较慢.
        point_j to camera_i with pt2d_{ij} and calc residuals
        :return:
        """
        n_cameras = len(self.nKFs)
        n_points = len(self.mMapPoints)
        camera_array = params[:n_cameras * 9].reshape((n_cameras, 9))
        points_3d = params[n_cameras * 9:].reshape((n_points, 3))
        points_proj = project(points_3d[self.point_indices], camera_array[self.camera_indices])
        return (points_proj - self.points_2d).ravel()

    def project_rot_matrix(self, params):
        """
        point_j to camera_i with pt2d_{ij} and calc residuals
        :return:
        """
        n_cameras = len(self.nKFs)
        n_points = len(self.mMapPoints)
        camera_array = params[:n_cameras * 9].reshape((n_cameras, 9))
        point_array = params[n_cameras * 9:].reshape((n_points, 3))
        error = np.empty(0)
        point_num = len(point_array)
        one_col = np.ones([point_num, 1])
        point_array_homo = np.append(point_array, one_col, axis=1)
        for i in range(len(self.nKFs)):
            point2d = self.get_point_2d(self.nKFs[i])
            pose_matrix = np.eye(4)
            phi = camera_array[i][:3]
            rho = camera_array[i][3:6]
            angle = np.linalg.norm(phi)
            if angle == 0:
                aplha = np.array([1, 0, 0])
            else:
                alpha = phi / angle
            pose_matrix = pose.PyLie.transform.rotation_matrix(angle, alpha)
            pose_matrix[:3, 3] = rho
            camera_point_homo = np.dot(pose_matrix, point_array_homo[self.point_dicts[i]].T)
            camera_points = camera_point_homo[:3, :]
            camera_points = camera_points[:2, :] / - camera_points[np.newaxis, 2, :]
            f = camera_array[i, 6]
            k1 = camera_array[i, 7]
            k2 = camera_array[i, 8]
            n = np.sum(camera_points ** 2, axis=0)
            r = 1 + k1 * n + k2 * n ** 2
            camera_points *= (r * f)[:, np.newaxis].T
            error_cur = camera_points.T - point2d
            error = np.append(error, error_cur.ravel())
        return error

    @staticmethod
    def random_partition(n, n_data):
        """return n random rows of data (and also the other len(data)-n rows)"""
        all_idxs = np.arange(n_data)
        np.random.shuffle(all_idxs)
        idxs1 = all_idxs[:n]
        idxs2 = all_idxs[n:]
        return idxs1, idxs2

    def fit_plane_ransac(self, iter_num, iterations, th, in_num, data, return_all=False, debug=False):
        """
        fit plane use ransac
        :param iter_num: 每次迭代中，采样的数量
        :param iterations: 迭代次数
        :param th: inlier 阈值
        :param in_num: 最小inlier数量
        :param data: 原始数据
        :param return_all: 是否返回所有
        :param debug: 显示每次迭代中数据
        :return: normal， plane 法向量
        """
        it = 0
        bestfit = None
        besterr = np.inf
        best_inlier_idxs = None
        while it < iterations:
            maybe_idxs, test_idxs = self.random_partition(iter_num, data.shape[0])
            maybeinliers = data[maybe_idxs, :]
            test_points = data[test_idxs]
            maybemodel = self.fit_plane(maybeinliers)
            test_err = self.fit_plane_error(maybemodel, test_points)
            also_idxs = test_idxs[test_err < th]  # select indices of rows with accepted points
            alsoinliers = data[also_idxs, :]
            if debug:
                print 'test_err.min()', test_err.min()
                print 'test_err.max()', test_err.max()
                print 'numpy.mean(test_err)', np.mean(test_err)
                print 'iteration %d:len(alsoinliers) = %d' % (
                    it, len(alsoinliers))
            if len(alsoinliers) > in_num:
                betterdata = np.concatenate((maybeinliers, alsoinliers))
                bettermodel = self.fit_plane(betterdata)
                better_errs = self.fit_plane_error(bettermodel, betterdata)
                thiserr = np.mean(better_errs)
                if thiserr < besterr:
                    bestfit = bettermodel
                    besterr = thiserr
                    best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))
            it += 1
        if bestfit is None:
            raise ValueError("did not meet fit acceptance criteria")
        if return_all:
            return bestfit, {'inliers': best_inlier_idxs}
        else:
            return bestfit

    @staticmethod
    def fit_plane(v_plane_points):
        """
        fit plane use svd，记过测试发现ransac的效果其实没有比svd的精度提高多少，但时间变慢了好多。
        :param v_plane_points: 原始数据
        :return: normal: plane normal
        """
        N = v_plane_points.shape[0]
        oneCol = np.ones([N, 1])
        v_plane_points = np.append(v_plane_points, oneCol, axis=1)
        _, _, V = np.linalg.svd(v_plane_points)
        vPlaneParams = V[3, :]
        return vPlaneParams

    @staticmethod
    def fit_plane_error(normal, data):
        """
        give plane normal and point to estimate the fit errors.
        :param normal: plane normal.
        :param data: original points data.
        :return: fit error
        """
        num = data.shape[0]
        ons_col = np.ones([num, 1])
        data = np.append(data, ons_col, axis=1)
        err = np.dot(normal, data.T)
        return err

    def homograph_error(self, vPlanePoints, vImagePoints, pose2Cam, camera):
        """
        use plane points to estimate the plane.
        use plane normal and pose to estimate the homograph.
        used the homograph matrix to calc the project error.
        """
        vPlaneParams = self.fit_plane(vPlanePoints)
        t = np.expand_dims(pose2Cam[:3, 3], axis=1)
        vNormal = np.expand_dims(vPlaneParams[:3], axis=0)
        H = pose2Cam[:3, :3] - np.dot(t, vNormal) / vPlaneParams[3]
        H = np.dot(camera, H)
        proj = np.dot(H, vPlanePoints.T)
        proj[:2, :] /= proj[2, :]
        return np.abs(vImagePoints.T - proj[:2, :]), H

    def reprj_error(self, world_points, image_points, pose, camera):
        """
        tranform the world points to the camera coordinates.
        project the camera points to the image plane and calc the errors.
        :param world_points: point coordinates in world.
        :param image_points: image coordinates in image plane.
        :param pose: camera pose in the world ordinates.
        :param camera: camera intrinsic parameters.
        :return: reprojection errors.
        """
        assert world_points.shape[0] == image_points.shape[0]
        assert pose.shape == (4, 4)
        point_num = world_points.shape[0]
        one_cols = np.ones([point_num, 1])
        world_points_homo = np.append(world_points, one_cols, axis=1)
        camera_points_homo = np.dot(pose, world_points_homo.T)
        camera_points = camera_points_homo[:3, :] / camera_points_homo[3, :]
        image_reprj = np.dot(camera, camera_points)
        image_reprj = image_reprj[:2, :] / image_reprj[2, :]
        return np.abs(image_reprj.T - image_points)
