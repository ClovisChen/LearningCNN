#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import least_squares


class SE3_XYZ_IMU():
    def __init__(self):
        self.poses = None
        self.points = None
        self.obs = None
        self.imus = None
        self.points_indice = None
        self.cameras_indice = None

    def __add__(self, other):
        self.poses = np.concatenate((self.poses, other.poses), axis=0)
        self.points = np.concatenate((self.points, other.points), axis=0)
        self.obs = np.concatenate((self.obs, other.obs), axis=0)
        self.points_indice = np.concatenate((self.points_indice, other.points_indice), axis=0)
        self.cameras_indice = np.concatenate((self.cameras_indice, other.cameras_indice), axis=0)
        self.imus = np.concatenate((self.imus, other.imus), axis=0)
