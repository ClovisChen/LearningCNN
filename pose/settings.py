#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

numPyrmaid = 3
RES_TH = 100.0
numFrameWindow = 8
minNumTracedPoint = 30
depth_scale = 1000.0
minDepth = 1.0
maxDepth = 20.0
GRID_SIZE = 40
LINE_GRID_SIZE = 200


lk_params = dict(winSize=(19, 19),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=1000,
                      qualityLevel=0.01,
                      minDistance=8,
                      blockSize=19)

green = (0, 255, 0)
red = (0, 0, 255)

Gravity = 9.810

gyrBiasRw2 = 2.0e-5 * 2.0e-5  # 陀螺仪的随机游走方差
accBiasRw2 = 5.0e-3 * 5.0e-3  # 加速度计随机游走方差
gyrMeasError2 = 1.7e-1 * 1.7e-1  # 陀螺仪的测量方差
accMeasError2 = 2.0e1 * 2.0e1  # 加速度计测量方差

scale_pyramid = 2
scale_factor = scale_pyramid ** np.arange(numPyrmaid)
level_sigma2 = scale_factor * scale_factor
inv_scale_factor = 1.0 / scale_factor
inv_level_sigma2 = 1.0 / level_sigma2

min_line_len = 200
max_line_direction_err = 1.0

bias_acc_prior=np.array((-0.025, 0.136, 0.075))
