#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pose.struct import *


def read_lines(data_dir, ret_float=True):
    with open(data_dir) as file:
        data = file.read()
        lines = data.split("\n")
        lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        if ret_float:
            return np.float64(lists)
        else:
            return lists


def test_measurement_matrix():
    data_root = '/home/bobin/data/line/VGG/Merton-College-I/'
    line_3d = data_root + 'l3d'
    line_2d = data_root + '001.lines'
    gt_proj_file = data_root + '001.P'
    match_file = data_root + 'nview-lines'
    x3d = read_lines(line_3d)
    x2d = read_lines(line_2d)
    gt_proj = read_lines(gt_proj_file)
    matches = read_lines(match_file)
    matches = np.int32(matches)[:, 0]

    np.set_printoptions(precision=6)

    cam, rot, trans = Plucker.decom_proj(gt_proj)
    print cam
    print rot
    print trans
    fxy = np.zeros((1, 4))
    fxy[0, 0] = fxy[0, 2] = cam[0, 0]
    fxy[0, 1] = fxy[0, 3] = cam[1, 1]

    cxy = np.zeros((1, 4))
    cxy[0, :2] = cxy[0, 2:] = cam[:2, 2]
    ptc = (x2d - cxy) / fxy
    proj = Plucker.fit_proj_matrix(x3d, ptc[matches, :])
    r1, r2, t1, t2 = Plucker.proj2rot_trans(proj)
    # print proj
    print 'gt rot and trans'
    print r1
    print r2
    print t1
    print t2


if __name__ == '__main__':
    test_measurement_matrix()
