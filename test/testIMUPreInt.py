#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pose.IMUPreInt as imu
import pose.PyLie.transform as tf
import pose.IMUPreInt as IMUPreInt
import data.euroc_reader as euroc


def test_imu_init(data_root):
    reader = euroc.euroc_data(data_root)
    reader.read_img_imu_gt()
    imu_stamp = reader.imu_data[:, 0]
    frame_index = np.arange(15) * 22 + 300
    frame_set = reader.gt_data[frame_index]
    start_time = frame_set[0, 0]
    end_time = frame_set[-1, 0]
    imu_index = (imu_stamp >= start_time) & (imu_stamp < end_time)
    imu_set = reader.imu_data[imu_index, :]
    est_imu_init = IMUPreInt.EstIMUInit(frame_set, imu_set)
    est_imu_init.preint()
    est_imu_init.optimize_bg()
    est_imu_init.est_gravity()
    est_imu_init.est_bias_acc()


def test_imu_preint(data_root):
    reader = euroc.euroc_data(data_root)
    reader.read_img_imu_gt()
    imu_data_0 = reader.imu_data[0]
    _gyro_0 = imu_data_0[1:4]
    _acc_0 = imu_data_0[4:]
    _time_0 = imu_data_0[0]
    imu_start = imu.IMUData(_gyro=_gyro_0, _acc=_acc_0, _time_stamp=_time_0)
    imu_preint = imu.IMUPreInt(imu_start)
    for i in range(1, 10):
        imu_data_i = reader.imu_data[i]
        _gyro_i = imu_data_i[1:4]
        _acc_i = imu_data_i[4:]
        _time_i = imu_data_i[0]
        _imu_data_ = imu.IMUData(_gyro=_gyro_i, _acc=_acc_i, _time_stamp=_time_i)
        imu_preint = imu_preint + _imu_data_
    print imu_preint.delta_rot
    print imu_preint.delta_pos
    print imu_preint.delta_speed


def test_svd_solve():
    A = np.floor(np.random.rand(40, 4) * 20 - 10)  # generating a np.random
    b = np.floor(np.random.rand(40, 1) * 20 - 10)  # system Ax=b

    U, s, V = np.linalg.svd(A, full_matrices=False)  # SVD decomposition of A

    # computing the inverse using pinv
    pinv = np.linalg.pinv(A)
    # computing the inverse using the SVD decomposition
    pinv_svd = np.dot(np.dot(V.T, np.linalg.inv(np.diag(s))), U.T)

    print "Inverse computed by lingal.pinv()\n", pinv
    print "Inverse computed using SVD\n", pinv_svd
    x = np.linalg.solve(A, b)  # solve Ax=b using np.linalg.solve

    xPinv = np.dot(pinv_svd, b)  # solving Ax=b computing x = A^-1*b

    # solving Ax=b using the equation above
    c = np.dot(U.T, b)  # c = U^t*b
    w = np.linalg.solve(np.diag(s), c)  # w = V^t*c
    xSVD = np.dot(V.T, w)  # x = V*w

    print "Ax=b solutions compared"
    print x.T
    print xSVD.T
    print xPinv.T


if __name__ == '__main__':
    data_root = '/home/bobin/data/euroc/MH_01_easy/'
    # imu_dir = data_root + 'imu0/data.csv'
    # gt_dir = data_root + 'state_groundtruth_estimate0/data.csv'

    test_imu_init(data_root)
    # test_svd_solve()
