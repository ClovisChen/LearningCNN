#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
from scipy import optimize
import scipy
import settings
import sophus


class Camera:
    """
    相机类，内参数，畸变参数，极线长度（双目），图像宽和高，
    """

    def __init__(self, fx, fy, cx, cy, w, h, d, bl):
        self.cx = cx
        self.cy = cy
        self.bl = bl
        self.d = d
        self.width = w
        self.height = h
        self.fx = fx
        self.fy = fy

    def project_2d_3d(self, uvd):
        """
        将图像点反投影到相机坐标系，利用当前的坐标和深度。
        :param uvd:
        :return:
        """
        u = uvd[0]
        v = uvd[1]
        d = uvd[2]
        x = d * (u - self.cx) / self.fx
        y = d * (v - self.cy) / self.fy
        z = d
        return [x, y, z]

    def get_intrinsic(self):
        """
        获取相机的内参数
        :return: camera intrinsic
        """
        return np.array([[self.fx, 0., self.cx],
                         [0., self.fy, self.cy],
                         [0., 0., 1.]])

    def project(self, point):
        """Project a 3D point in camera coordinates to the image plane."""
        x, y, z = point
        u = x / z * self.fx + self.cx
        v = x / z * self.fy + self.cy
        return [u, v]


class PyrCamera:
    """
    构造相机的金字塔内参数
    """

    def __init__(self, _camera, num_pyr):
        self.camera = dict()
        self.camera[0] = _camera
        for lvl in range(num_pyr):
            self.camera[lvl] = Camera(_camera.fx * 0.5 ** lvl,
                                      _camera.fy * 0.5 ** lvl,
                                      (_camera.cx + 0.5) * 0.5 ** lvl - 0.5,
                                      (_camera.cy + 0.5) * 0.5 ** lvl - 0.5,
                                      _camera.width >> lvl,
                                      _camera.height >> lvl,
                                      _camera.d,
                                      _camera.bl)

    def project_2d_3d(self, uvd, lvl):
        return self.camera[lvl].project_2d_3d(uvd)

    def project(self, point, lvl):
        return self.camera[lvl].project(point)


class Frame:
    def __init__(self, timeStamp=None, left_image=None, right_image=None, depth_image=None, camera=None):
        """
        帧结构，构造时间戳，frameid，左右图像，深度图像，相机
        :param timeStamp: 时间戳
        :param left_image: 左图像
        :param right_image: 右图像
        :param depth_image: 深度图 像
        :param camera: 金字塔 相机
        """
        self.mTimeStamp = timeStamp
        self.mLeftImage = left_image
        self.mRightImage = right_image
        self.mDepthImage = depth_image

        self.height = list()
        self.width = list()

        # 构造图像金字塔，左眼图像，梯度x，y，mag，深度图像
        self.numPyrmaid = 4
        self.border = 30
        self.mLeftPyr = list()
        self.mGradXPyr = list()
        self.mGradYPyr = list()
        self.mGradMagPyr = list()
        self.mDepthPyr = list()

        # 构造结构数据，2dpoint，3d点，gftt点，梯度点。
        self.points = None
        self.mPyrPoint = dict()
        self.mMapPoints = dict()
        self.mFeatures = dict()

        # 构造运动数据，相机的位姿，位姿真值
        self.mPose2World = sophus.SE3()
        self.mGTPose2World = sophus.SE3()

        # 标定相关数据
        self.GradSize = (40, 20, 10, 5)
        self.cam = camera

        # 开始进行一些每帧 要做的工作
        if left_image is None or depth_image is None or camera is None:
            return

        # assert isinstance(camera.camera, dict)
        self.left_pyr()

    def set_pose(self, pose2world):
        assert pose2world.shape == (4, 4)
        self.mPose2World = sophus.SE3(pose2world.T.flatten())

    def set_gt_pose(self, pose2world):
        assert pose2world.shape == (4, 4)
        self.mGTPose2World = sophus.SE3(pose2world.T.flatten())

    def assign_features_grid(self):
        """
        划分图像的网格，检查当前帧中的特征，网格中存在特征的mask白色，没有特征的mask黑色
        :return:
        """
        height, width = self.mLeftImage.shape
        num_grid_rows = int(height / settings.GRID_SIZE)
        num_grid_cols = int(width / settings.GRID_SIZE)
        # grid = np.ones([num_grid_rows, num_grid_cols], dtype=np.uint8)
        mask = 255 * np.ones([height, width], dtype=np.uint8)

        for pt in self.points[:, 0]:
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

    def left_pyr(self):
        """
        构造图像金字塔数据左眼图像，梯度图像
        :return:
        """
        self.mLeftPyr.append(self.mLeftImage)
        gradx = cv2.Sobel(self.mLeftImage, cv2.CV_64F, 1, 0, ksize=3)
        grady = cv2.Sobel(self.mLeftImage, cv2.CV_64F, 0, 1, ksize=3)
        self.mGradXPyr.append(np.uint8(np.absolute(gradx)))
        self.mGradYPyr.append(np.uint8(np.absolute(grady)))
        self.width.append(self.mLeftImage.shape[1])
        self.height.append(self.mLeftImage.shape[0])
        self.mGradMagPyr.append(np.uint8(np.absolute((np.sqrt(gradx * gradx + grady * grady)))))
        self.mDepthPyr.append(self.mDepthImage)

        for lvl in range(1, settings.numPyrmaid):
            self.mLeftPyr.append(cv2.pyrDown(self.mLeftPyr[lvl - 1]))
            self.width.append(self.width[lvl - 1] / 2)
            self.height.append(self.height[lvl - 1] / 2)
            gradx = cv2.Sobel(self.mLeftPyr[lvl - 1], cv2.CV_64F, 1, 0, ksize=3)
            grady = cv2.Sobel(self.mLeftPyr[lvl - 1], cv2.CV_64F, 0, 1, ksize=3)
            self.mGradXPyr.append(np.uint8(np.absolute(gradx)))
            self.mGradYPyr.append(np.uint8(np.absolute(grady)))
            self.mGradMagPyr.append(np.uint8(np.absolute(np.sqrt(gradx * gradx + grady * grady))))
            self.mDepthPyr.append(cv2.pyrDown(self.mDepthPyr[lvl - 1]))

    def mark_points(self):
        mark_image = self.mLeftImage.copy()
        for pt in self.points[:, 0]:
            cv2.circle(mark_image, (pt[0], pt[1]), 2, 255)
        return mark_image

    def mark_pyr_points(self, lvl):
        mark_image = self.mLeftImage.copy()
        mark_image = np.expand_dims(mark_image, axis=2)
        mark_image = np.repeat(mark_image, 3, axis=2)
        point_lvl = self.mPyrPoint[lvl]
        for pt in point_lvl.itervalues():
            cv2.circle(mark_image, (pt[1], pt[0]), 2, (0, 0, 255))
        return mark_image

    @staticmethod
    def huber(r, delta=20):
        """
        calc huber loss
        :param delta:
        :param r:
        :return: loss
        """
        if delta < 0:
            return np.inf
        elif abs(r) < delta:
            return r * r * 0.5
        else:
            return delta * (abs(r) - delta * 0.5)

    def calcHb(self, target_fm, lvl):
        """
        对当前帧构造hessian points用于计算误差，高斯牛顿优化
        将当前帧（host frame）中的点投影目标帧（target freame），优化目标帧的位姿，当前帧的位姿，计算线性化误差，
        :param target_fm:
        :return:
        """
        if not self.mPyrPoint.has_key(lvl):
            return
        inv_pose = self.mPose2World.matrix()
        R = inv_pose[:3, :3]
        t = inv_pose[:3, 3]
        inv_pose[:3, 3] = - np.dot(R.T, t)
        inv_pose[:3, :3] = R.T
        relative_pose = np.dot(inv_pose, target_fm.mPose2World.matrix())
        H_acc = np.zeros([6, 6])
        b_acc = np.zeros([6, 1])
        resAcc = 0
        for pt in self.mPyrPoint[lvl].itervalues():
            ptc_host = self.cam.project_2d_3d(pt, lvl)
            pt_target = np.dot(relative_pose, np.append(ptc_host, 1))
            uxy_target = self.cam.project(pt_target[:3] / pt_target[3], lvl)
            gray_host = self.mLeftImage[pt[1], pt[0]]
            gray_target = target_fm.get_pixel_value(uxy_target[0], uxy_target[1], lvl)
            if gray_target is None:
                continue
            gradx = self.mGradXPyr[lvl][pt[1], pt[0]]
            grady = self.mGradYPyr[lvl][pt[1], pt[0]]
            # res = self.huber(gray_host - gray_target)
            res = gray_host - gray_target
            loss = self.huber(res)
            H, b = self.linearizeOplus(loss, res, ptc_host[:3], g=[gradx, grady], lvl=lvl)
            H_acc += H
            b_acc += b
            resAcc += res
        return H_acc, b_acc, resAcc

    def inc_pose(self, inc):
        self.mPose2World = self.mPose2World * sophus.SE3.exp(inc)

    def get_pixel_value(self, x, y, lvl=0):
        if (x + 1) >= self.cam.camera[lvl].width or x < 0 or y < 0 or (y + 1) >= self.cam.camera[lvl].height:
            return None
        image = self.mLeftPyr[lvl]
        y_int = int(y)
        x_int = int(x)
        left_top = image[y_int, x_int]
        right_top = image[y_int, x_int + 1]
        left_bottom = image[y_int + 1, x_int]
        right_bottom = image[y_int + 1, x_int + 1]
        xx = x - x_int
        yy = y - y_int
        return (1 - xx) * (1 - yy) * left_top + \
               xx * (1 - yy) * right_top + \
               (1 - xx) * yy * left_bottom + \
               xx * yy * right_bottom

    def linearizeOplus(self, loss, res, pt, g, lvl):
        """
        计算线性化误差，根据光度学误差，计算雅克比矩阵，Hessian matrix。
        :param res:
        :param pt:
        :param g:
        :return:
        """
        jaccobian = np.zeros([2, 6])
        [x, y, z] = pt
        fx = self.cam.camera[lvl].fx
        fy = self.cam.camera[lvl].fy
        z_inv = 1.0 / z
        z_inv2 = z_inv * z_inv
        jaccobian[0, 3] = fx * z_inv
        jaccobian[1, 4] = fy * z_inv
        jaccobian[0, 5] = - fx * x * z_inv2
        jaccobian[1, 5] = - fy * y * z_inv2

        jaccobian[0, 0] = - fx * x * y * z_inv2
        jaccobian[1, 0] = - fy - fy * y * y * z_inv2
        jaccobian[0, 1] = fx + fx * x * x * z_inv2
        jaccobian[1, 1] = fy * x * y * z_inv2
        jaccobian[0, 2] = -fx * y * z_inv
        jaccobian[1, 2] = fy * x * z_inv
        J = res * np.dot(g, jaccobian)
        J = np.expand_dims(J, 1)
        sigma = 1.
        s2 = 1.0 / sigma
        w = 1.0 / (1.0 + loss * loss * s2)
        # w = 1.0
        H = np.dot(J, J.T) * w * w
        b = - J * loss * w
        return H, b

    def point_select_grid(self):
        """"
        ## make grids ,then find point with maximium gradient mangtitude in every grid.
        """
        for k in range(settings.numPyrmaid):
            count = 0
            point_lvl = dict()
            for i in range((self.height[k] - 2 * self.border) / self.GradSize[k]):
                for j in range((self.width[k] - 2 * self.border) / self.GradSize[k]):
                    pty0 = self.border + i * self.GradSize[k]
                    pty1 = self.border + (i + 1) * self.GradSize[k]
                    ptx0 = self.border + j * self.GradSize[k]
                    ptx1 = self.border + (j + 1) * self.GradSize[k]
                    pt_pos = np.argmax(self.mGradMagPyr[k][pty0:pty1, ptx0:ptx1])
                    pty = pty0 + pt_pos / self.GradSize[k]
                    ptx = ptx0 + pt_pos % self.GradSize[k]
                    d = float(self.mDepthPyr[k][pty, ptx]) / settings.depth_scale
                    if settings.minDepth < d < settings.maxDepth:
                        point_lvl[count] = [ptx, pty, d]
                        count += 1
            self.mPyrPoint[k] = point_lvl

    def add_feature(self, feature):
        """
        add the feature to the feature dict.
        :param feature:
        :return:
        """
        key = len(self.mFeatures)
        self.mFeatures[key] = feature

    @staticmethod
    def left_right_trace(frame, back_threshold=1.0):
        p0 = frame.points
        p1, st, err = cv2.calcOpticalFlowPyrLK(frame.mLeftImage, frame.mRightImage, p0, None, **(settings.lk_params))
        p0r, st, err = cv2.calcOpticalFlowPyrLK(frame.mRightImage, frame.mLeftImage, p1, None, **(settings.lk_params))
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        status = d < back_threshold
        return p1, status


if __name__ == '__main__':
    image = np.random.random([4, 4])
    print image
    frame = Frame(left_image=image)
    frame.mLeftPyr.append(image)
    # print frame.getPixelValue(2.00, 2.9999, lvl=0)
    # Inc = np.array([[0.07543147, 0.61393189, -0.78574661, 1.3405],
    #                 [0.9970987, -0.03837025, 0.06574118, 0.6266],
    #                 [0.01021131, -0.78842588, -0.61504501, 1.6575],
    #                 [0., 0., 0., 1.]])
    #
    # print Inc
    # T = sophus.SE3(Inc.T.flatten())
    # # T *= Inc
    # # T.setRotationMatrix()
    # # INC = sophus.SE3.exp(Inc)
    #
    # print T.matrix()
