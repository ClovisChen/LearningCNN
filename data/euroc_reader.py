#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import yaml
from yaml_read import *
import cv2
import os


class euroc_data():
    def __init__(self, data_root):
        self.img_list = list()
        self.imu_data = None
        self.gt_data = None
        self.data_root = data_root
        self.imu_dir = data_root + 'imu0/data.csv'
        self.gt_dir = data_root + 'state_groundtruth_estimate0/data.csv'
        self.left_img_dir = 'cam0/data.csv'
        self.right_img_dir = 'cam1/data.csv'
        self.intrinsic_left = data_root + 'cam0/sensor.yaml'
        self.intrinsic_right = data_root + 'cam1/sensor.yaml'
        self.cam_left = dict()
        self.cam_right = dict()
        self.params_file = '/Euroc/Euroc.yaml'
        self.params = dict()
        self.M1l = None
        self.M2l = None
        self.M1r = None
        self.M2r = None

    @staticmethod
    def read_euroc_data(data_dir, ret_float=True):
        with open(data_dir) as file:
            data = file.read()
            lines = data.split("\n")
            lists = [[v.strip() for v in line.split(",") if v.strip() != ""] for line in lines if
                     len(line) > 0 and line[0] != "#"]
            if ret_float:
                return np.float64(lists)
            else:
                return lists

    def read_imu_data(self):
        self.imu_data = self.read_euroc_data(self.imu_dir)

    def read_gt_data(self):
        self.gt_data = self.read_euroc_data(self.gt_dir)

    def read_img_list(self):
        left_img_stamp = self.read_euroc_data(self.data_root + self.right_img_dir, ret_float=False)
        right_img_stamp = self.read_euroc_data(self.data_root + self.right_img_dir, ret_float=False)
        img_list = list()
        for left, right in zip(left_img_stamp, right_img_stamp):
            img_left = self.data_root + 'cam0/data/' + left[1]
            img_right = self.data_root + 'cam1/data/' + right[1]
            img_list.append((img_left, img_right))
        self.img_list = img_list

    def read_img_imu_gt(self):
        self.read_imu_data()
        self.read_gt_data()
        self.read_img_list()
        self.read_params()

    def read_intrinsic(self):
        with open(self.intrinsic_left, 'r') as f:
            self.cam_left = yaml.load(f)

        with open(self.intrinsic_right, 'r') as f:
            self.cam_right = yaml.load(f)

        K_l = np.eye(3)
        K_l[0, 0] = self.cam_left['intrinsics'][0]
        K_l[1, 1] = self.cam_left['intrinsics'][1]
        K_l[0, 2] = self.cam_left['intrinsics'][2]
        K_l[1, 2] = self.cam_left['intrinsics'][3]

        D_l = self.cam_left['distortion_coefficients']

        K_r = np.eye(3)
        K_r[0, 0] = self.cam_right['intrinsics'][0]
        K_r[1, 1] = self.cam_right['intrinsics'][1]
        K_r[0, 2] = self.cam_right['intrinsics'][2]
        K_r[1, 2] = self.cam_right['intrinsics'][3]

        D_r = self.cam_right['distortion_coefficients']
        print D_l, D_r
        print K_l, K_r

    def read_params(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + self.params_file, 'r') as f:
            self.params = yaml.load(f)

            K_l = self.params["LEFT.K"]
            K_r = self.params["RIGHT.K"]

            P_l = self.params["LEFT.P"]
            P_r = self.params["RIGHT.P"]

            R_l = self.params["LEFT.R"]
            R_r = self.params["RIGHT.R"]

            D_l = self.params["LEFT.D"]
            D_r = self.params["RIGHT.D"]

            rows_l = self.params["LEFT.height"]
            cols_l = self.params["LEFT.width"]
            rows_r = self.params["RIGHT.height"]
            cols_r = self.params["RIGHT.width"]

            self.M1l, self.M2l = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l[:3, :3], (cols_l, rows_l), cv2.CV_32F)
            self.M1r, self.M2r = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r[:3, :3], (cols_r, rows_r), cv2.CV_32F)
