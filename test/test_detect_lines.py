#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import time
from pylsd import lsd
from pose.struct import LineStruct
from pose.track import Tracker
min_line_len = 50


def read_file_list(filename):
    with open(filename) as fp:
        data = fp.read()
        lines = data.split("\n")
        lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        return lists


if __name__ == '__main__':
    data_root = '/media/bobin/Seagate/data/slam/iacas/indoor/video_2/'
    file_list = 'file_apple_indoor.txt'
    image_names = read_file_list(data_root + file_list)

    for cnt, (left, right) in enumerate(image_names):
        src = cv2.imread(data_root + left, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        tic = time.clock()
        lines = LineStruct.detect_lines(gray)
        # Tracker.detect_gftt_feature(gray)
        toc = time.clock()
        print 'lsd cost time %f ms' % (toc - tic)
        for i in xrange(lines.shape[0]):
            pt1 = (int(lines[i, 0]), int(lines[i, 1]))
            pt2 = (int(lines[i, 2]), int(lines[i, 3]))
            width = lines[i, 4]
            # cv2.line(src, pt1, pt2, (0, 255, 255), int(np.ceil(width / 2)))
            cv2.line(src, pt1, pt2, (0, 255, 255), 1)
        if cnt != 0:
            traced_pt, status = LineStruct.lk_line_end(lines, gray, last_gray)
            for pt, st, line in zip(traced_pt, status, lines):
                if st:
                    pt1 = (int(line[0]), int(line[1]))
                    pt2 = (int(pt[0, 0]), int(pt[0, 1]))
                    # cv2.line(src, pt1, pt2, (255, 0, 0), 1)
        else:
            last_src = src
        last_gray = gray
        # continue

        v1,v2 = LineStruct.calc_vanishing_pt(lines)
        x_axis_ori = v1[:2]
        y_axis_ori = v2[:2]

        h, w = gray.shape
        pt_ori = np.array([w / 2, h / 2], dtype=int)
        axis_len = 100
        # y_axis_len = np.array([0, 100])
        pt_x_axis = pt_ori + axis_len * (x_axis_ori)
        pt_y_axis = pt_ori + axis_len * (y_axis_ori)

        pt_x_axis = np.int32(pt_x_axis)
        pt_y_axis = np.int32(pt_y_axis)

        # pt_x_axis = np.array([200, 200], dtype=int)
        cv2.line(src, (pt_ori[0], pt_ori[1]), (pt_x_axis[0], pt_x_axis[1]), (255, 0, 0), 3)
        cv2.line(src, (pt_ori[0], pt_ori[1]), (pt_y_axis[0], pt_y_axis[1]), (255, 0, 0), 3)
        # cv2.line(src, (pt_ori[0], pt_ori[1]), (pt_z_axis[0], pt_z_axis[1]), (255, 0, 0), 3)
        # cv2.line(src, pt_ori, pt_y_axis, (255, 0, 0))
        # print V

        cv2.imshow('lines', src)
        # if last_src is not None:
        # cv2.imshow('last', last_src)
        last_src = src
        cv2.waitKey(10)
