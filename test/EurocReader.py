#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def read_imu_data(data_dir):
    with open(data_dir) as file:
        data = file.read()
        lines = data.split("\n")
        lists = [[v.strip() for v in line.split(",") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        return np.float64(lists)


def read_ground_truth(filename):
    with open(filename) as file:
        data = file.read()
        lines = data.split("\n")
        lists = [[v.strip() for v in line.split(",") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        return np.float64(lists)


def read_images_lists(filenames):
    with open(filenames) as file:
        data = file.read()
        lines = data.split("\n")
        lists = [[v.strip() for v in line.split(",") if v.strip() != ""] for line in lines if
                 len(line) > 0 and line[0] != "#"]
        return lists