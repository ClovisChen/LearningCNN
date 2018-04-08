#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import yaml
from yaml_read import *
import cv2
import os
from glob import glob


class KITTIData:
    def __init__(self, data_root, sequence):
        self.data_root = data_root + '/dataset/sequences/%.2d/' % sequence
        self.img_list = list()
        self.gt_data = None

        self.gt_dir = data_root + 'poses/%.2d.txt' % sequence
        self.left_img_dir = self.data_root + 'image_0/'
        self.right_img_dir = self.data_root + 'image_1/'
        if 0 <= sequence <= 2:
            self.params_file = '/KITTI/KITTI00-02.yaml'
        elif sequence == 3:
            self.params_file = '/KITTI/KITTI03.yaml'
        else:
            self.params_file = '/KITTI/KITTI04-12.yaml'
        self.params = dict()
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.bf = None

    @staticmethod
    def read_kitti_data(data_dir, ret_float=True):
        with open(data_dir) as file:
            data = file.read()
            lines = data.split("\n")
            lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                     len(line) > 0 and line[0] != "#"]
            if ret_float:
                return np.float64(lists)
            else:
                return lists

    @staticmethod
    def read_calib_file(filepath, cid=2):
        """Read in a calibration file and parse into a dictionary."""
        with open(filepath, 'r') as f:
            C = f.readlines()

        def parseLine(L, shape):
            data = L.split()
            data = np.array(data[1:]).reshape(shape).astype(np.float32)
            return data

        proj_c2p = parseLine(C[cid], shape=(3, 4))
        proj_v2c = parseLine(C[-1], shape=(3, 4))
        filler = np.array([0, 0, 0, 1]).reshape((1, 4))
        proj_v2c = np.concatenate((proj_v2c, filler), axis=0)
        return proj_c2p, proj_v2c

    def read_gt_data(self):
        self.gt_data = self.read_kitti_data(self.gt_dir)

    def read_img_list(self):
        N = len(glob(self.left_img_dir + '/*.png'))
        left_name = [self.left_img_dir + '%.6d.png' % n for n in range(N)]
        right_name = [self.right_img_dir + '%.6d.png' % n for n in range(N)]
        return left_name, right_name

    def read_params(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(dir_path + self.params_file, 'r') as f:
            self.params = yaml.load(f)
            self.fx = self.params["Camera.fx"]
            self.fy = self.params['Camera.fy']
            self.cx = self.params['Camera.cx']
            self.cy = self.params['Camera.cy']
            self.bf = self.params['Camera.bf']
