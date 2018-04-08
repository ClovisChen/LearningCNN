#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import settings
import numpy as np


class Camera:
    def __init__(self, fx, fy, cx, cy, bf):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.bf = bf


class Pose:
    def __init__(self, rot, trans):
        self.rot = rot
        self.trans = trans


class FrameLK:
    def __init__(self, left=None, right=None, camera=None):
        self.points = None
        self.left = left
        self.right = right
        self.pose = None
        self.point_index = None
        self.camera = camera
        self.rot_trans = None
        self.key_frame = False

    def assign_features_grid(self):
        if self.points is None:
            return
        height, width = self.left.shape
        num_grid_rows = int(height / settings.GRID_SIZE)
        num_grid_cols = int(width / settings.GRID_SIZE)
        # grid = np.ones([num_grid_rows, num_grid_cols], dtype=np.uint8)
        mask = 255 * np.ones((height, width), dtype=np.uint8)

        for pt in self.points:
            x = int(pt[0] / settings.GRID_SIZE)
            y = int(pt[1] / settings.GRID_SIZE)
            if x >= num_grid_cols:
                x = num_grid_cols - 1
            if y >= num_grid_rows:
                y = num_grid_rows - 1
            # grid[y, x] = False
            ptx0 = x * settings.GRID_SIZE
            ptx1 = (x + 1) * settings.GRID_SIZE
            pty0 = y * settings.GRID_SIZE
            pty1 = (y + 1) * settings.GRID_SIZE
            if ptx1 >= width:
                ptx1 = width - 1
            if pty1 >= height:
                pty1 = height - 1
            mask[pty0:pty1, ptx0:ptx1] = 0
        return mask

    def left_right_trace(self, points, point_3d_start_id, back_threshold=1.0):
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
        point_3d_id = point_3d_start_id + np.arange(len(point_3d))
        if self.points is not None:
            self.points = np.concatenate((self.points, traced_pt), axis=0)
            self.point_index = np.append(self.point_index, point_3d_id, axis=0)
        else:
            self.points = traced_pt
            self.point_index = point_3d_start_id + point_3d_id
        return point_3d

    def detect_gftt_feature(self, mask):
        feature_params = settings.feature_params.copy()
        feature_params['mask'] = mask
        pts = cv2.goodFeaturesToTrack(self.left, **(feature_params))
        pts = pts.reshape((-1, 2))
        return pts

    def draw_points(self, img):
        for pt in self.points:
            cv2.circle(img, (pt[0], pt[1]), 2, 255)

    def trace_last(self, last, back_threshold=1.0):
        current_pt, st, err = cv2.calcOpticalFlowPyrLK(last.left, self.left, last.points, None, **(settings.lk_params))
        last_pt_r, st, err = cv2.calcOpticalFlowPyrLK(self.left, last.left, current_pt, None, **(settings.lk_params))
        d = abs(last.points - last_pt_r).reshape(-1, 2).max(-1)
        status = d < back_threshold
        self.points = current_pt[status].reshape((-1, 2)).copy()
        self.point_index = last.point_index[status]

    def estimate_current_pose(self, points_3d, camera):
        if len(self.point_index) > 15:
            return FrameLK.pose_estimate(self.points, points_3d[self.point_index], camera)
        else:
            return None

    @staticmethod
    def pose_estimate(pts_2d, pts_3d, camera):
        cam_mat = np.eye(3)
        cam_mat[0, 0] = camera.fx
        cam_mat[1, 1] = camera.fy
        cam_mat[0, 2] = camera.cx
        cam_mat[1, 2] = camera.cy
        dist = np.zeros((5, 1))
        pt3 = pts_3d[:, :, np.newaxis].astype(np.float32)
        pt2 = pts_2d[:, :, np.newaxis]
        rot, trans, inliers = cv2.solvePnPRansac(pt3, pt2, cam_mat, dist)
        return rot.ravel(), trans.ravel()
