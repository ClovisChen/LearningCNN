#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

sys.path.insert(0, '../data')
sys.path.insert(0, '../utils')
import cv2
import Frame

import numpy  as np
import sophus
# import pose_utils
# import transform
import scipy
import settings
from Queue import Queue
import matplotlib.pyplot as plt
import pylsd as lsd
plt.ion()


class Tracker:
    def __init__(self, camera):
        """
        tracker 初始化，定义了变量当前帧和上一帧，由于估计帧间的变化。
        定义了是否初始化，涉及提取特征。
        定义一个帧序列，用于存储帧的集合。
        :param camera:
        """
        self.mCurrentFrame = None
        self.mLastFrame = None
        self.use_ransac = True
        self.initialized = False
        self.Camera = camera
        self.NeedKeyFrame = True
        self.mCurrentLvl = 1
        self.FrameArray = dict()
        self.FrameWindow = list()
        self.mMapPoints = dict()

    def need_key_frame(self, traced_pt_nums):
        if traced_pt_nums < settings.minNumTracedPoint:
            self.NeedKeyFrame = True
        else:
            self.NeedKeyFrame = False

    def motion_model(self):
        inc_mat = self.mLastFrame.mPose2World.inverse() * self.mCurrentFrame.mPose2World
        inc = sophus.SE3.log(inc_mat)
        motion_list = dict()
        motion_list[0] = inc
        rho = inc[3:]
        theta = inc[:3]
        motion_list[0] = [rho, theta]
        motion_list[1] = [0.5 * rho, theta]
        motion_list[2] = [0.25 * rho, theta]
        motion_list[3] = [2 * rho, theta]
        motion_list[4] = [4 * rho, theta]
        motion_list[5] = [-rho, theta]
        motion_list[6] = [-0.5 * rho, theta]
        motion_list[7] = [-0.25 * rho, theta]
        motion_list[8] = [-2 * rho, theta]
        motion_list[9] = [-4 * rho, theta]
        motion_list[10] = [rho, theta]
        motion_list[11] = [rho, 0.5 * theta]
        motion_list[12] = [rho, 0.25 * theta]
        motion_list[13] = [rho, 2 * theta]
        motion_list[14] = [rho, 4 * theta]
        motion_list[15] = [rho, - theta]
        motion_list[16] = [rho, -0.5 * theta]
        motion_list[17] = [rho, -0.25 * theta]
        motion_list[18] = [rho, -2 * theta]
        motion_list[19] = [rho, -4 * theta]
        return motion_list

    @staticmethod
    def draw_str(dst, (x, y), s):
        cv2.putText(dst, s, (x + 1, y + 1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=2, lineType=cv2.CV_AA)
        cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

    def checked_trace(self, img0, img1, p0, back_threshold=1.0):
        """
        use lk of to trace points between image 0 and image 1.
        :param img0:
        :param img1:
        :param p0:
        :param back_threshold:
        :return: return traced points and their status
        """
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **(settings.lk_params))
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **(settings.lk_params))
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        status = d < back_threshold
        if len(p1.flatten()) < 2 * settings.minNumTracedPoint:
            self.NeedKeyFrame = True
        else:
            self.NeedKeyFrame = False
        return p1, status

    @staticmethod
    def detect_gftt_feature(frame):
        """
        detect feature in frame, if there already have points in grid, do not detect it.
        :param frame:
        :return: None
        """
        Mask = frame.assign_features_grid()
        feature_params = settings.feature_params.copy()
        feature_params['mask'] = Mask
        pts = cv2.goodFeaturesToTrack(frame.mLeftImage, **(feature_params))
        if pts is not None:
            frame.points = np.append(frame.points, pts, axis=0)

    def lk_track_init(self, frame):
        """
        init lk tracker, detect gftt features in last frame.
        :param frame:
        :return: None
        """
        frame.points = cv2.goodFeaturesToTrack(frame.mLeftImage, **(settings.feature_params))
        if frame.points is not None:
            self.initialized = True
        self.mCurrentFrame = frame

    def lk_track(self, frame):
        """
        use lk optical flow to estimate the homograph between last frame and current frame.
        :param frame:
        :return: homograph matrix
        """
        if not self.initialized:
            self.lk_track_init(frame)
            return
        self.mLastFrame = self.mCurrentFrame
        self.mCurrentFrame = frame
        if self.mLastFrame.points is not None:
            p2, trace_status = self.checked_trace(self.mLastFrame.mLeftImage, self.mCurrentFrame.mLeftImage,
                                                  self.mLastFrame.points)
            self.mCurrentFrame.points = p2[trace_status].copy()

            self.detect_gftt_feature(self.mCurrentFrame)
            mark_image = self.mCurrentFrame.mark_points()
            return mark_image

    def calc_homography(self, p0, p1, visualize=True):
        """
        通过点对应计算单应矩阵，另外如果需要验证，那么使用可视化，使用一个平面的wrap表示
        :param p0:
        :param p1:
        :param visualize:
        :return:
        """
        H, status = cv2.findHomography(p0, p1, (0, cv2.RANSAC)[self.use_ransac], 10.0)
        if visualize is True:
            vis = self.mCurrentFrame.mLeftImage.copy()
            h, w = self.mCurrentFrame.mLeftImage.shape[:2]
            overlay = cv2.warpPerspective(self.mLastFrame.mLeftImage, H, (w, h))
            vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)

            for (x0, y0), (x1, y1), good in zip(p0[:, 0], p1[:, 0], status[:, 0]):
                if good:
                    cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
                cv2.circle(vis, (x1, y1), 2, (settings.red, settings.green)[good], -1)
            self.draw_str(vis, (20, 20), 'track count: %d' % len(p1))
            if self.use_ransac:
                self.draw_str(vis, (20, 40), 'RANSAC')
                # cv2.imshow('lk_homography', vis)
        return H

    def calc_fundamental(self, p0, p1, verify=False):
        """
        通过点对应计算之间的fundamental matrix
        :param p0:
        :param p1:
        :param verify: 如果为True，那么通过gt值计算，进行比较。
        :return:
        """
        F, status = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC, 3, 0.99)
        U, _, V = np.linalg.svd(F)
        R90 = transform.rotation_matrix(np.pi * 0.5, [0, 0, 1])[:3, :3]
        R = np.dot(np.dot(U, R90), V)
        t_hat = np.dot(np.dot(U, R90), V)

        if verify == True:
            T1 = self.mLastFrame.mGTPose2World.matrix()
            T2 = self.mCurrentFrame.mGTPose2World.matrix()
            T12 = np.dot(np.linalg.inv(T1), T2)
            R12 = T12[:3, :3]
            t12 = T12[:3, 3]
            t12hat = sophus.SO3.hat(t12)
            essential = np.dot(t12hat, R12)
            K = self.Camera.get_intrinsic()
            print K
            E = np.dot(np.dot(K.T, F), K)
            fundamental = np.dot(np.dot(np.linalg.inv(K).T, essential), np.linalg.inv(K))
            print F, '\n', fundamental
            # print  E, '\n', essential
            print F / fundamental
            print K.T, np.linalg.inv(K.T), np.linalg.inv(K).T
        return R, t_hat

    def trace_points(self, H):
        """
        use initialized homograph estimation to trace more points in current frame to last frame.
        :param H:
        :return: traced point in current frame.
        """
        if H is None:
            return
        points_last_frame = dict()
        for lvl in range(settings.numPyrmaid - 1, self.mCurrentLvl - 1, -1):
            point_lvl = list()
            for ptCurrent in self.mCurrentFrame.mPyrPoint[lvl].itervalues():
                if ptCurrent is None:
                    break
                pt_homo_current = ptCurrent[:2]
                pt_homo_current.append(1)
                pt_last = np.dot(H, pt_homo_current)
                pt_last /= pt_last[2]
                point_lvl.append(pt_last[:2])
            points_last_frame[lvl] = point_lvl
        return points_last_frame

    def mark_track_points(self, H):
        """
        这个mark point没啥用啊，每帧提取的点并不进行鉴别，重投影是没有意义的。
        :param H:
        :return:
        """
        if self.mCurrentFrame is None:
            return
        mark_image = self.mCurrentFrame.mLeftImage.copy()
        mark_image = np.expand_dims(mark_image, axis=2)
        mark_image = np.repeat(mark_image, 3, axis=2)
        point_lvl = self.mLastFrame.mPyrPoint[0]
        for pt in point_lvl.itervalues():
            cv2.circle(mark_image, (pt[1], pt[0]), 2, (0, 0, 255))
        point_last_frame = self.trace_points(H)
        if point_last_frame is None:
            return
        for pt in point_last_frame[0]:
            cv2.circle(mark_image, (int(pt[1]), int(pt[0])), 2, (0, 255, 0))
        cv2.imshow("homo transform", mark_image)

    def pose_gaussian_newton(self):
        """
        金字塔从高层到当前层，根据窗口中的每帧的点，计算帧的hessian和residual，迭代计算当前帧的位姿，如果residual小于一定阈值，那么停止优化
        :return:
        """
        for lvl in range(settings.numPyrmaid - 1, self.mCurrentLvl - 1, -1):
            H_acc = np.zeros([6, 6])
            b_acc = np.zeros([6, 1])
            res_acc = 0
            for i in range(5):
                for frame in self.FrameWindow:
                    H, b, res = frame.calcHb(self.mCurrentFrame, lvl)
                    H_acc += H
                    b_acc += b
                    res_acc += res
                    # print 'lvl ', lvl, 'res ',res
                K = scipy.linalg.cho_factor(H_acc)
                inc = scipy.linalg.cho_solve(K, b_acc)
                # print inc.flatten()
                if max(np.abs(inc)) > 0.2:
                    continue
                self.mCurrentFrame.inc_pose(0.5 * inc)

    def insert_frame(self, frame, frame_id=0):
        """
        将当前要处理的帧划分网格，进行LK光流跟踪，对没有特征点的地方提取gftt特征。估计与上一帧的位姿变换，然后采样图像中的点，使用直接法估计位姿。三角花这些点，进行稠密重建。
        :param frame: 要插入的帧
        :param frame_id: 要插入帧对应ID
        :return: None
        """
        if frame is None:
            return
        self.FrameArray[frame_id] = frame
        if frame_id is 0:
            # self.LKTrackInit(frame)
            self.mCurrentFrame = frame
            self.mCurrentFrame.mPose2World = self.mCurrentFrame.mGTPose2World
        else:
            self.mLastFrame = self.mCurrentFrame
            self.mCurrentFrame = frame
            # self.mCurrentFrame.mPose2World = self.mCurrentFrame.mGTPose2World
            self.mCurrentFrame.mPose2World = self.mLastFrame.mPose2World
            # self.LKTrack(frame)
            self.pose_gaussian_newton()
        if self.NeedKeyFrame:
            while len(self.FrameWindow) >= settings.numFrameWindow:
                self.FrameWindow.pop(0)
            self.FrameWindow.append(frame)
        print "est", sophus.SE3.log(self.mCurrentFrame.mPose2World).flatten(), \
            "gt", sophus.SE3.log(self.mCurrentFrame.mGTPose2World).flatten()
        return self.mCurrentFrame.mPose2World.matrix()

    def add_map_point(self, point):
        if point is not None:
            current_num = len(self.mMapPoints)
            key = current_num
            self.mMapPoints[key] = point

    @staticmethod
    def compute_stereo_matches(frame):
        right_pts, trace_status = Frame.Frame.left_right_trace(frame)
        p1 = right_pts[trace_status].copy()
        p0 = frame.points[trace_status].copy()
        good_idx = p0[:, 0] < p1[:, 0] & (p0[:, 1] - p1[:, 1] < 5)
        disp = - np.ones(len(p0))
        disp[good_idx] = p0[good_idx, 0] - p1[good_idx, 0]
        return np.concatenate((p0, disp), axis=1)

    @staticmethod
    def detect_lines(image):
        lsd.lsd()