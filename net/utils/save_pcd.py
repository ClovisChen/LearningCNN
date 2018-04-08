#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pcl
import numpy as np


def save_npy_pcd(npy_filename, pcd_out):
    points = np.load(npy_filename)
    if points.shape[1] == 3:
        cloud = pcl.PointCloud(np.float32(points))
    else:
        cloud = pcl.PointCloud(np.float32(points[:, :3]))
    cloud.to_file(pcd_out)


if __name__ == '__main__':
    data_root = '/home/bobin/code/net/geometric/learningReloc/net/data/model/output/hobot/'
    npy_file = data_root + 'map.npy'
    pcd_file = data_root + 'map.pcd'
    save_npy_pcd(npy_file, pcd_file)
