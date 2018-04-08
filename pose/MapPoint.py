#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Feature:
    def __init__(self, pixel=[0, 0], map_point_key=-1):
        """
        2d pixel
        3d map point key
        """
        self.mPixel = pixel
        self.mMapPointKey = map_point_key

    def set_pixel(self, pixel):
        self.mPixel = pixel

    def set_map_point(self, map_point_key=-1):
        self.mMapPointKey = map_point_key


class MapPoint:
    def __init__(self, frame, pixel_d, lvl=0):
        """
        :param frame: 最开始看到该点的帧，看到该点的帧保存于observations中，
        :param pixel_d:
        :param lvl:
        """
        self.mPixel = pixel_d[:2]
        self.depth = pixel_d[2]
        ptc = frame.cam.camera[lvl].project_2d_3d(pixel_d)
        self.mCamPos = ptc
        ptc.append(1)
        ptc = np.array(ptc)
        pose = frame.mPose2World.matrix()
        Ow = pose[:3, 3]
        pos_homo = np.dot(pose, ptc)
        self.mWorldPos = pos_homo[:3] / pos_homo[3]
        self.mNormalVector = self.mWorldPos - Ow
        self.mNormalVector = self.mNormalVector / np.linalg.norm(self.mNormalVector)
        self.observations = dict()

    def __init__(self, map_point):
        self.mWorldPos = map_point

    @staticmethod
    def triangulate(pose_0, feat_0, pose_1, feat_1, cam):
        cx = cam[0, 2]
        cy = cam[1, 2]
        fx = cam[0, 0]
        fy = cam[1, 1]
        xn0 = np.array([(feat_0[0] - cx) / fx, (feat_0[1] - cy) / fy, 1])
        xn1 = np.array([(feat_1[0] - cx) / fx, (feat_1[1] - cy) / fy, 1])
        Rcw0 = pose_0[:3, :3]
        tcw0 = pose_0[:3, 3]
        Rcw1 = pose_1[:3, :3]
        tcw1 = pose_1[:3, 3]
        ray0 = Rcw0 * xn0
        ray1 = Rcw1 * xn1

        cos_parallax_rays = ray0.dot(ray1) / (np.linalg.norm(ray0) * np.linalg.norm(ray1))
        if 0 < cos_parallax_rays < 1:
            A = np.zeros((4, 4))
            A[1, :] = xn0[0] * pose_0[2, :] - pose_0[0, :]
            A[2, :] = xn0[1] * pose_0[2, :] - pose_0[1, :]
            A[3, :] = xn1[0] * pose_1[2, :] - pose_0[0, :]
            A[4, :] = xn1[1] * pose_1[2, :] - pose_0[1, :]
            _, _, V = np.linalg.svd(A)
            x3D = V[3, :3] / V[3, 3]

            z0 = Rcw0[0, :].dot(x3D) + tcw0[0]
            z1 = Rcw1[0, :].dot(x3D) + tcw1[1]
            if z0 < 0 or z1 < 0:
                return None
            x0 = Rcw0[0, :].dot(x3D) + tcw0[0]
            y0 = Rcw0[1, :].dot(x3D) + tcw0[1]
            u0 = fx * x0 / z0 + cx
            v0 = fy * y0 / z0 + cy
            err_x = u0 - feat_0[0]
            err_y = v0 - feat_0[1]
            err_0 = err_x * err_x + err_y * err_y

            x1 = Rcw0[0, :].dot(x3D) + tcw0[0]
            y1 = Rcw0[1, :].dot(x3D) + tcw0[1]
            u1 = fx * x1 / z1 + cx
            v1 = fy * y1 / z1 + cy
            err_x = u1 - feat_0[0]
            err_y = v1 - feat_0[1]
            err_1 = err_x * err_x + err_y * err_y
            return x3D, err_0, err_1
        else:
            return None


if __name__ == '__main__':
    import sophus
    import Frame

    d = [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]
    camera = Frame.Camera(517.3, 516.5, 318.6, 255.3, 640, 480, d, None)
    pyr_camera = Frame.PyrCamera(camera, 4)
    frame = Frame.Frame(camera=pyr_camera)
    # frame.mPose2World = sophus.SE3()
    frame.mPose2World = sophus.SE3.rotY(np.pi / 4) * frame.mPose2World.trans(0.1, 0.2, 0.3)
    # print np.dot((sophus.SE3.rotX(np.pi / 4).matrix())[:3, :3], frame.mPose2World.trans(0.1, 0.2, 0.3).matrix()[:3, 3])
    pixel_depth = [318.6, 255.3, 10]
    point = MapPoint(frame, pixel_depth, 0)
    print point.mPixel
    print point.mWorldPos
    print point.mNormalVector
    print point.mCamPos
    print frame.mPose2World

    pointset = dict()
    pointset[0] = point
    print pointset[0].mPixel
