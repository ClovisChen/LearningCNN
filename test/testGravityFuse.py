#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pose.IMUPreInt as imu
import pose.PyLie.transform as tf
import pose.IMUPreInt as IMUPreInt
import pose.struct as line_struct
import data.euroc_reader as euroc
import cv2


def test_fuse_gravity(data_root):
    reader = euroc.euroc_data(data_root)
    reader.read_img_imu_gt()

    for left, right in reader.img_list:
        left_color = cv2.imread(left, cv2.IMREAD_COLOR)
        left_img = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
        right_color = cv2.imread(right, cv2.IMREAD_COLOR)
        right_img = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)
        left_un = cv2.remap(left_img, reader.M1l, reader.M2l, cv2.INTER_LINEAR)
        right_un = cv2.remap(right_img, reader.M1r, reader.M2r, cv2.INTER_LINEAR)

        lines = line_struct.detect_lines(left_un)
        line_struct.draw_lines(left_color, lines)
        img = np.concatenate((left_color, right_color), axis=1)
        cv2.imshow('left right', img)
        cv2.waitKey(10)
    est_imu_init = IMUPreInt.EstIMUInit()


if __name__ == '__main__':
    data_root = '/home/bobin/data/euroc/MH_01_easy/'
    test_fuse_gravity(data_root)
