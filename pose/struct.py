#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import lsd
import settings
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class LineStruct:
    def __init__(self, line_params=None, lines=None):
        """
        :param line_params:
            # 1. a
            # 2. b,
            # 3. c,
            # 4. sort_index,
            # 5. line seg length,
            # 6. long index
        :param lines:
            # 1. x0
            # 2. y0
            # 3. x1
            # 4. y1
            # 5. line width
        """
        self.line_params = line_params
        self.lines = lines

    @staticmethod
    def fit_lines(line_segs, simple=True):
        x0 = line_segs[:, 0]
        x1 = line_segs[:, 2]
        y0 = line_segs[:, 1]
        y1 = line_segs[:, 3]
        delta_x = x0 - x1
        delta_y = y0 - y1

        a = delta_y
        b = - delta_x
        c = x0 * y1 - y0 * x1
        a = np.expand_dims(a, axis=1)
        b = np.expand_dims(b, axis=1)
        c = np.expand_dims(c, axis=1)
        line_param = np.concatenate((a, b), axis=1)
        line_param = np.concatenate((line_param, c), axis=1)

        if simple:
            line_seg_lenth = np.sqrt(delta_x ** 2 + delta_y ** 2)
            sort_index = np.argsort(line_seg_lenth)[::-1]

            long_index = line_seg_lenth > settings.min_line_len

            sort_index = np.expand_dims(sort_index, axis=1)
            line_seg_lenth = np.expand_dims(line_seg_lenth, axis=1)
            long_index = np.expand_dims(long_index, axis=1)
            line_param = np.concatenate((line_param, sort_index), axis=1)
            line_param = np.concatenate((line_param, line_seg_lenth), axis=1)
            line_param = np.concatenate((line_param, long_index), axis=1)
        return line_param

    @staticmethod
    def detect_lines(gray):
        lines = lsd.detect_line_segments(gray.astype(np.float64), scale=0.5, density_th=0.6)
        return lines

    @staticmethod
    def draw_lines(src, lines):
        for i in xrange(lines.shape[0]):
            pt1 = (int(lines[i, 0]), int(lines[i, 1]))
            pt2 = (int(lines[i, 2]), int(lines[i, 3]))
            width = lines[i, 4]
            cv2.line(src, pt1, pt2, (0, 0, 255), int(np.ceil(width / 2)))

    @staticmethod
    def sample_in_line(lines):
        pt_set = list()
        for line in lines:
            pt_sub_set = list()
            length = int(line[4])
            start = np.array([line[0], line[1]])
            end = np.array([line[2], line[3]])
            pt_num = length / 10
            if pt_num == 0:
                continue
            for pt_idx in range(pt_num):
                part = pt_idx * 1.0 / pt_num
                pt = np.int32(part * start + (1 - part) * end)
                pt_sub_set.append(pt)
            pt_set.append(pt_sub_set)

    @staticmethod
    def trace_lines_points(cur_img, next_img, pt_lines, back_th):
        p1, st, err = cv2.calcOpticalFlowPyrLK(cur_img, next_img, pt_lines, None, **(settings.lk_params))
        p0r, st, err = cv2.calcOpticalFlowPyrLK(next_img, cur_img, p1, None, **(settings.lk_params))
        d = abs(pt_lines - p0r).reshape(-1, 2).max(-1)
        status = d < back_th
        return p1, status

    @staticmethod
    def cluster_theta(lines):
        theta = lines[:, :2]
        theta_norm = np.linalg.norm(theta, axis=1)
        theta /= np.expand_dims(theta_norm, axis=1)
        skew_grid = np.zeros(20, dtype=int)
        skew_grid_index = [None] * 20
        skew_inv_grid_index = [None] * 20
        skew_inv_grid = np.zeros(20, dtype=int)
        for cnt, t in enumerate(theta):
            if abs(t[0]) < 0.707:
                skew = t[0] / t[1]
                grid_index = int(10 * skew) + 10
                skew_grid[grid_index] += 1
                if skew_grid_index[grid_index] is None:
                    skew_grid_index[grid_index] = list()
                skew_grid_index[grid_index].append(cnt)
            else:
                skew_inv = t[1] / t[0]
                grid_inv_index = int(10 * skew_inv) + 10
                skew_inv_grid[grid_inv_index] += 1
                if skew_inv_grid_index[grid_inv_index] is None:
                    skew_inv_grid_index[grid_inv_index] = list()

                skew_inv_grid_index[grid_inv_index].append(cnt)
        skew_max = np.argmax(skew_grid)
        skew_inv_max = np.argmax(skew_inv_grid)
        c1_lines = np.array(skew_grid_index[skew_max])
        c2_lines = np.array(skew_inv_grid_index[skew_inv_max])
        c1_lines_weight = lines[c1_lines, :3]
        c2_lines_weight = lines[c2_lines, :3]
        _, _, v1 = np.linalg.svd(c1_lines_weight)
        var_pt1 = v1[2, :]
        _, _, v2 = np.linalg.svd(c2_lines_weight)
        var_pt2 = v2[2, :]
        return var_pt1, var_pt2

    @staticmethod
    def calc_vanishing_pt(lines):
        line_params = LineStruct.fit_lines(lines)
        return LineStruct.cluster_theta(line_params)

    @staticmethod
    def lk_line_end(lines, image0, image1, back_threshold=1.0):
        start_pts = np.float32(lines[:, :2])
        start_pts = np.expand_dims(start_pts, axis=1)
        p1, st, err = cv2.calcOpticalFlowPyrLK(image0, image1, start_pts, None, **(settings.lk_params))
        p0r, st, err = cv2.calcOpticalFlowPyrLK(image1, image0, p1, None, **(settings.lk_params))
        d = abs(start_pts - p0r).reshape(-1, 2).max(-1)
        status = d < back_threshold
        return p1, status

    @staticmethod
    def line_in_gravity(lines, gravity):
        err = lines.dot(gravity)
        return err < settings.max_line_direction_err, err

    @staticmethod
    def refine_gravity_direction(lines, gravity):
        err = lines.dot(gravity)
        weight = np.expand_dims(1.0 / err, axis=1)
        _, _, V = np.linalg.svd(lines * weight)
        print V[2, :]


class StructFrame:
    def __init__(self, left=None, right=None, camera=None):
        self.left = left
        self.right = right
        self.camera = camera
        self.pose = np.eye(4)

    def set_pose(self, _pose):
        self.pose = _pose

    def left_right_line_triangulate(self, lines, line_start_id,  back_threshold=1.0):
        num_line = len(lines)
        points = lines[:, :2]

        points = np.concatenate((points, lines[:, 2:4]), axis=0)
        right_pt, st, err = cv2.calcOpticalFlowPyrLK(self.left, self.right, points, None, **(settings.lk_params))
        left_pt_r, st, err = cv2.calcOpticalFlowPyrLK(self.right, self.left, right_pt, None, **(settings.lk_params))
        d = abs(points - left_pt_r).reshape(-1, 2).max(-1)
        row_d = abs(points[:, 1] - right_pt[:, 1])
        status = (d < back_threshold) & (row_d <= 5)
        disparity = points[:, 0] - right_pt[:, 0]
        max_d = (self.camera.fx + self.camera.fy) / 2.0
        min_d = 0
        status &= disparity > min_d
        status &= disparity < max_d
        depth = self.camera.bf / disparity

        traced_pt = points[status]
        traced_pt_d = depth[status]

        rot = self.pose[:3, :3]
        trans = self.pose[:3, 3, np.newaxis]
        cxy = np.array([[self.camera.cx, self.camera.cy]])
        fxy = np.array([[self.camera.fx, self.camera.fy]])

        point_c_z = np.expand_dims(traced_pt_d, axis=1)
        point_c_xy = (traced_pt - cxy) / fxy * point_c_z
        point_c = np.concatenate((point_c_xy, point_c_z), axis=1)
        point_3d = (rot.dot(np.array(point_c).T) + trans).T
        point_3d_id = line_start_id + np.arange(len(point_3d))

        line_3d = point_3d[:num_line, :]
        line_3d = np.concatenate((line_3d, point_3d[num_line:]), axis=1)
        return line_3d

        if self.points is not None:
            self.points = np.concatenate((self.points, traced_pt), axis=0)
            self.point_index = np.append(self.point_index, point_3d_id, axis=0)
        else:
            self.points = traced_pt
            self.point_index = line_start_id + point_3d_id
        return point_3d


class Plucker:
    def __init__(self):
        pass

    @staticmethod
    def plucker_matrix(x0, x1):
        plu_mat_right = x1 - x0
        plu_mat_left = np.cross(x0, x1)
        plu_mat = np.concatenate((plu_mat_left, plu_mat_right), axis=1)
        return plu_mat

    @staticmethod
    def plucker_transform(rot, trans):
        t_mat = np.zeros(6)
        t_mat[:3, :3] = rot
        t_mat[3:, 3:] = rot
        t_mat[3:, :3] = np.cross(trans, rot)
        return t_mat

    @staticmethod
    def measurement_matrix(x3d_ls, x2d_ls):
        num = len(x3d_ls)
        row_index = np.arange(num)
        a = x2d_ls[:, 0]
        b = x2d_ls[:, 1]
        c = x2d_ls[:, 2]
        mat = np.zeros((2 * num, 18))

        for col in range(6):
            mat[row_index, 3 * col] = c * x3d_ls[:, col]
            mat[row_index, 3 * col + 2] = -a * x3d_ls[:, col]
            mat[row_index + num, 3 * col + 1] = c * x3d_ls[:, col]
            mat[row_index + num, 3 * col + 2] = -b * x3d_ls[:, col]
        return mat

    @staticmethod
    def fit_proj_matrix(x3d_ls, x2d):
        x2d_ls = LineStruct.fit_lines(x2d, False)
        x3d_plu = Plucker.plucker_matrix(x3d_ls[:, :3], x3d_ls[:, 3:])
        measurement_mat = Plucker.measurement_matrix(x3d_plu, x2d_ls)
        _, _, V = np.linalg.svd(measurement_mat)
        proj_mat_lines = V[-1, :]
        return np.reshape(proj_mat_lines, (6, 3)).T

    @staticmethod
    def decom_proj(proj):
        cam, rot, trans, a1, a2, a3, a4 = cv2.decomposeProjectionMatrix(proj)
        return cam, rot, trans[:3] / trans[3]

    @staticmethod
    def proj2rot_trans(proj_mat):
        rot = proj_mat[:, :3]
        rot_det = np.linalg.det(rot)
        if rot_det < 0:
            s = -(1.0 / -rot_det) ** (1.0 / 3)
        else:
            s = (1.0 / rot_det) ** (1.0 / 3)
        proj_scale = s * proj_mat

        rot_t = proj_scale[:, 3:]
        U, S, V = np.linalg.svd(rot_t, full_matrices=True)
        if np.linalg.det(U) < 0:
            U *= -1
        if np.linalg.det(V) < 0:
            V *= -1

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rot1 = U.dot(W).dot(V)
        rot2 = U.dot(W.T).dot(V)

        avg_sg = 0.5 * (S[0] + S[1])
        Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
        t_xa = avg_sg * V.T.dot(Z).dot(V)
        t_xb = avg_sg * V.T.dot(Z.T).dot(V)
        t1 = t_xa
        t2 = t_xb
        return rot1, rot2, t1, t2

    @staticmethod
    def triangulate(proj0, line0, proj1, line1):
        proj = np.concatenate((proj0, proj1), axis=0)
        u, s, v = np.linalg.svd(proj)
        line = np.append(line0, line1)
        c = np.dot(u.T, line)
        w = np.linalg.solve(np.diag(s), c)
        x = np.dot(v.T, w)
        return x


class LineBundleAdjustment:
    POINT_RESIDUAL = 0
    LINE_RESIDUAL = 1

    def __init__(self, method):
        self.resisual_method = method

    @staticmethod
    def residual_point(params, n_proj, n_lines, projs_index, line_index, x2d, pt_index):
        projs = params[:3 * n_proj, :]
        x3dlines = params[3 * n_proj:, :]
        assert len(x3dlines) == n_lines
        a = x2d[:, 0]
        b = x2d[:, 1]
        c = x2d[:, 2]

        res1 = np.sum(projs[3 * projs_index, :] * x3dlines[line_index, :], axis=1)
        res2 = np.sum(projs[3 * projs_index + 1, :] * x3dlines[line_index, :], axis=1)
        res3 = np.sum(projs[3 * projs_index + 2, :] * x3dlines[line_index, :], axis=1)

        residual = a * res1[pt_index] + b * res2[pt_index] + res3[pt_index]
        return residual.ravel()

    @staticmethod
    def residual_line(params, n_proj, n_lines, projs_index, line_index, x2dline):
        projs = params[:3 * n_proj, :]
        x3dlines = params[3 * n_proj:, :]
        assert len(x3dlines) == n_lines
        a = x2dline[:, 0]
        b = x2dline[:, 1]
        c = x2dline[:, 2]

        res1 = np.sum(projs[3 * projs_index, :] * x3dlines[line_index, :], axis=1)
        res2 = np.sum(projs[3 * projs_index + 1, :] * x3dlines[line_index, :], axis=1)
        res3 = np.sum(projs[3 * projs_index + 2, :] * x3dlines[line_index, :], axis=1)

        resiual1 = c * res1 - a * res3
        resiual2 = c * res2 - b * res3

        resiual = np.append(resiual1, resiual2, axis=0)
        return resiual.ravel()

    @staticmethod
    def optimize_line(projs, lines, projs_index, line_index, x2dline, sparsity):
        n_projs = len(projs) / 3
        n_lines = len(lines)

        res = least_squares(LineBundleAdjustment.residual_line, jac_sparsity=sparsity, verbose=2, x_scale='jac',
                            ftol=1e-4, method='trf', args=(n_projs, n_lines, projs_index, line_index, x2dline))
        return res.x

    @staticmethod
    def optimize_point(projs, lines, projs_index, line_index, x2d, pt_index, sparsity):
        n_projs = len(projs) / 3
        n_lines = len(lines)

        res = least_squares(LineBundleAdjustment.residual_point, jac_sparsity=sparsity, verbose=2, x_scale='jac',
                            ftol=1e-4, method='trf', args=(n_projs, n_lines, projs_index, line_index, x2d, pt_index))
        return res.x

    @staticmethod
    def point_sparsity(projs_index, line_index, pt_index):
        n_points = len(pt_index)
        n_projs = len(projs_index)
        n_lines = len(line_index)

        dof_proj = 6
        dof_line = 6
        n = n_projs * dof_proj + n_lines * dof_line

        A = lil_matrix((n_points, n), dtype=int)
        for s in range(dof_proj):
            A[pt_index, projs_index * 6 + s] = 1

        for s in range(dof_line):
            A[pt_index, n_projs * 6 + line_index * 6 + s] = 1

        return A
