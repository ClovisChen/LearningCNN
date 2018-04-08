#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import yaml
import glob
import contextlib


# A yaml constructor is for loading from a yaml node.
# This is taken from @misha 's answer: http://stackoverflow.com/a/15942429
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    if mapping["cols"] > 1:
        mat.resize(mapping["rows"], mapping["cols"])
    else:
        mat.resize(mapping["rows"], )
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    if mat.ndim > 1:
        mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    else:
        mapping = {'rows': mat.shape[0], 'cols': 1, 'dt': 'd', 'data': mat.tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)


yaml.add_representer(np.ndarray, opencv_matrix_representer)


def make_file_name_list():
    data_root = '/home/chen-tian/data/data/KITTI/odom/'
    left_file_names = glob.glob(data_root + 'dataset/sequences/00/image_2/*.png')
    right_file_names = glob.glob(data_root + 'dataset/sequences/00/image_3/*.png')
    left_file_names.sort()
    right_file_names.sort()
    with open('file_names.txt', 'w') as fp:
        # count = 0
        for l, r in zip(left_file_names, right_file_names):
            # if (count % 10) == 0:
            local_path_left = '/'.join(l.split('/')[-5:])
            local_path_right = '/'.join(r.split('/')[-5:])
            fp.write(local_path_left)
            fp.write(' ')
            fp.write(local_path_right)
            fp.write('\n')
            # count += 1


def load_kitti_trajectory(filename):
    with open(filename) as fp:
        data = fp.read()
        lines = data.split('\n')
        lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        trajectory = np.zeros((len(lists), 12))
        for i, item in enumerate(lists):
            trajectory[i, :] = np.float64(item)
        return trajectory


def load_velodyne_trajectory(filename):
    with open(filename) as fp:
        data = fp.read()
        lines = data.split("\n")
        lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        trajectory = np.zeros((len(lists), 8))
        for i, item in enumerate(lists):
            timestamp = item[0]
            trajectory[i, 0] = float(item[0])
            trajectory[i, 1:] = np.float64(item[2:])
        return trajectory


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


if __name__ == '__main__':
    make_file_name_list()
    # trajectory = '/home/bobin/data/hobot/data-sh/sensor_20170803-154153_/tra.txt'
    # tra = load_velodyne_trajectory(trajectory)
    # with printoptions(precision=20, suppress=False):
    #     print len(tra)
    #     print tra[0]
