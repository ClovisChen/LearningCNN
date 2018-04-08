#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pose.PyLie.transform as tf
import pose.settings as settings
import pose.PyLie.Quaternion as Quat
import pose.PyLie.SO3 as SO3
from scipy.optimize import least_squares


def normalize(v):
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v / norm


def normalize_quat(q):
    if q[0] < 0:
        q *= -1
    return normalize(q)


def normalize_rot(mat):
    q = Quat.mat2quat(mat)
    return Quat.quat2mat(normalize_quat(q))


class IMUData():
    gyro_Meas_cov = np.eye(3) * settings.gyrMeasError2
    acc_meas_cov = np.eye(3) * settings.accMeasError2

    def __init__(self, _gyro, _acc, _time_stamp):
        self.gyro = _gyro
        self.acc = _acc
        self.time_stamp = _time_stamp
        self.gravity = np.array([0, 0, settings.Gravity])


class IMUPreInt():
    def __init__(self, init_pose, init_time, _acc_bias, gyro_bias):
        self.delta_rot = np.eye(3)
        self.delta_speed = np.zeros(3)
        self.delta_pos = np.zeros(3)
        self.init_time = init_time
        self.time_stamp = init_time
        self.init_pose = init_pose
        self.cur_pose = init_pose
        self.bias_acc = _acc_bias
        self.bias_gyro = gyro_bias

        self.jac_p_bias_g = np.zeros((3, 3))
        self.jac_p_bias_a = np.zeros((3, 3))
        self.jac_v_bias_g = np.zeros((3, 3))
        self.jac_v_bias_a = np.zeros((3, 3))
        self.jac_r_bias_g = np.zeros((3, 3))

        self.conv_p_v_phi = np.zeros((9, 9))
        self.delta_time = 0

    def __add__(self, other):
        delta_t = other.time_stamp - self.time_stamp
        delta_t /= 1e9
        omega_delta_t = other.gyro * delta_t
        theta = np.linalg.norm(omega_delta_t)
        rot_vec = omega_delta_t / theta
        delta_rot_inc = tf.rotation_matrix(theta, rot_vec)[:3, :3]
        self.delta_rot = self.delta_rot.dot(delta_rot_inc)
        self.delta_rot = normalize_rot(self.delta_rot)
        self.delta_speed += self.delta_rot.dot(other.acce) * delta_t
        delta_t2 = delta_t * delta_t
        self.delta_pos = self.delta_speed * delta_t + 0.5 * self.delta_rot * other.acc * delta_t2
        self.time_stamp = other.time_stamp
        return self

    def pre_int(self, acc, gyro, dt):
        omega_dt = gyro * dt
        theta = np.linalg.norm(omega_dt)
        if theta < np.finfo(float).eps:
            delta_rot_inc = np.eye(3)
        else:
            rot_vec = omega_dt / theta
            delta_rot_inc = tf.rotation_matrix(theta, rot_vec)[:3, :3]
        self.delta_rot = self.delta_rot.dot(delta_rot_inc)
        self.delta_rot = normalize_rot(self.delta_rot)
        self.delta_speed += self.delta_rot.dot(acc) * dt
        dt2 = dt * dt
        self.delta_pos = self.delta_speed * dt + 0.5 * self.delta_rot.dot(acc) * dt2
        self.time_stamp += dt

    def update(self, acc, gyro, dt, calc_cov=False):
        dt2 = dt * dt
        omega_dt = gyro * dt
        jac_right = SO3.SO3.right_jac(omega_dt)
        dR = SO3.exp_so3(omega_dt)
        I3x3 = np.eye(3)
        acc_h = SO3.skew_matrix(acc)
        if calc_cov:
            A = np.eye(9)
            A[6:9, 6:9] = dR.T
            A[3:6, 6:9] = - self.delta_rot.dot(acc_h) * dt
            A[0:3, 6:9] = -0.5 * self.delta_rot.dot(acc_h) * dt
            A[0:3, 3:6] = I3x3 * dt
            Bg = np.zeros((9, 3))
            Bg[6:9, 0:3] = jac_right * dt

            Ca = np.zeros((9, 3))
            Ca[3:6, :] = -self.delta_rot * dt
            Ca[0:3, :] = 0.5 * self.delta_rot * dt2
            self.conv_p_v_phi = A.dot(self.conv_p_v_phi).dot(A.T)
            self.conv_p_v_phi += Bg.dot(IMUData.gyro_Meas_cov).dot(Bg.T) + Ca.dot(IMUData.acc_meas_cov).dot(Ca.T)

        self.jac_p_bias_a += self.jac_v_bias_a * dt - 0.5 * self.delta_rot * dt2
        self.jac_p_bias_g += self.jac_v_bias_g * dt - 0.5 * self.delta_rot.dot(acc_h) * self.jac_r_bias_g * dt2
        self.jac_v_bias_g += - self.delta_rot.dot(acc_h) * self.jac_r_bias_g * dt
        self.jac_v_bias_a += - self.delta_rot * dt
        self.jac_r_bias_g = dR.T.dot(self.jac_r_bias_g) - jac_right * dt
        # self.pre_int(acc, gyro, dt)

        self.delta_pos += self.delta_speed * dt + 0.5 * self.delta_rot.dot(acc) * dt2
        self.delta_speed += self.delta_rot.dot(acc) * dt
        self.delta_rot = self.delta_rot.dot(dR)
        self.delta_rot = normalize_rot(self.delta_rot)
        self.time_stamp += dt
        self.delta_time += dt

    def int_pose(self):
        delta_pose = np.eye(4)
        delta_pose[:3, :3] = self.delta_rot
        delta_pose[:3, 3] = self.delta_pos
        cur_pose = self.init_pose.dot(delta_pose)
        return cur_pose

    def set_current_pose(self, pose):
        self.cur_pose = pose


class EstIMUInit():
    def __init__(self, frame_set, imu_set):
        self.frames = frame_set
        self.imus = imu_set

        self.frame_size = len(frame_set)
        self.imu_preint_array = [None] * (self.frame_size - 1)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.gravity = np.zeros(3)

    def optimize_bg(self):
        res = least_squares(self.residual_bg, self.bg, verbose=2, x_scale='jac', ftol=1e-4,  # jac=self.jacobian,
                            method='trf')
        print res.x
        self.bg = res.x

    def residual_bg(self, data):
        num = len(self.imu_preint_array)
        error = np.zeros((num, 3))
        for cnt, imu_pre_int in enumerate(self.imu_preint_array):
            d_rot_b_ij = imu_pre_int.delta_rot
            jac_rot_bg = imu_pre_int.jac_r_bias_g
            rot_w_bi = imu_pre_int.init_pose[:3, :3]
            rot_w_bj = imu_pre_int.cur_pose[:3, :3]
            # conv_p_v_phi = imu_pre_int.conv_p_v_phi
            d_rot_bg = SO3.exp_so3(jac_rot_bg.dot(data))
            tmp1 = d_rot_b_ij.dot(d_rot_bg)
            tmp2 = tmp1.T.dot(rot_w_bi.T)
            error_mat = tmp2.dot(rot_w_bj)
            error[cnt, :] = SO3.log_so3(error_mat)
        return error.ravel()

    def jacobian(self, data):
        num = len(self.imu_preint_array)
        jac_oplus_xi = np.zeros((3 * num, 3))
        for cnt, imu_pre_int in enumerate(self.imu_preint_array):
            d_rot_b_ij = imu_pre_int.delta_rot
            jac_rot_bg = imu_pre_int.jac_r_bias_g
            rot_w_bi = imu_pre_int.init_pose[:3, :3]
            rot_w_bj = imu_pre_int.cur_pose[:3, :3]
            tmp1 = d_rot_b_ij.T.dot(rot_w_bi.T)
            error_mat = tmp1.dot(rot_w_bj)
            error = SO3.log_so3(error_mat)
            jac_left_inv = SO3.SO3.left_jac_inv(error)
            jac_oplus_xi[3 * cnt:3 * (cnt + 1), :] = - jac_left_inv.dot(jac_rot_bg)
        return jac_oplus_xi

    def preint(self):
        time_stamp = self.imus[:, 0]
        for i in range(self.frame_size - 1):
            cur_frame = self.frames[i, :]
            next_frame = self.frames[i + 1, :]
            start_time = cur_frame[0]
            end_time = next_frame[0]
            cur_quat = cur_frame[4:8]
            cur_trans = cur_frame[1:4]
            cur_pose = tf.quaternion_matrix(cur_quat)
            cur_pose[:3, 3] = cur_trans

            next_quat = next_frame[4:8]
            next_trans = next_frame[1:4]
            next_pose = tf.quaternion_matrix(next_quat)
            next_pose[:3, 3] = next_trans

            frame_between = (time_stamp >= start_time) & (time_stamp < end_time)
            imu_set = self.imus[frame_between]
            cur_time = start_time
            imu_preint = IMUPreInt(cur_pose, start_time, 0, 0)
            imu_preint.set_current_pose(next_pose)

            for imu_item in imu_set:
                dt = imu_item[0] - cur_time
                cur_time = imu_item[0]
                acc = imu_item[4:8]
                gyro = imu_item[1:4]
                imu_preint.update(acc - self.ba, gyro - self.bg, dt / 1e9)
            dt = end_time - cur_time
            imu_preint.update(acc - self.ba, gyro - self.bg, dt / 1e9)
            self.imu_preint_array[i] = imu_preint

    def est_bias_acc(self):
        A = np.zeros((3 * (self.frame_size - 2), 3))
        B = np.zeros((3 * (self.frame_size - 2), 1))
        for i in range(self.frame_size - 2):
            cur_frame = self.imu_preint_array[i]
            next_frame = self.imu_preint_array[i + 1]
            nnext_pose = self.frames[i + 2]
            cur_n_dt = cur_frame.delta_time
            n_nn_dt = next_frame.delta_time

            cur_rot = cur_frame.init_pose[:3, :3]
            next_rot = next_frame.init_pose[:3, :3]
            cur_pos = cur_frame.init_pose[:3, 3]
            next_pos = next_frame.init_pose[:3, 3]
            nnext_pos = nnext_pose[1:4]

            cur_n_delta_pos = cur_frame.delta_pos
            n_nn_delta_pos = next_frame.delta_pos
            cur_n_delta_speed = cur_frame.delta_speed
            cur_n_jac_p_ba = cur_frame.jac_p_bias_a
            cur_n_jac_v_ba = cur_frame.jac_v_bias_a
            n_nn_jac_p_ba = next_frame.jac_p_bias_a

            Lambda = 0.5 * (cur_n_dt ** 2 * n_nn_dt + cur_n_dt * n_nn_dt * n_nn_dt) * np.eye(3).dot(self.gravity)
            Phi = next_rot.dot(n_nn_jac_p_ba) * cur_n_dt
            Phi += cur_rot.dot(cur_n_jac_p_ba) * n_nn_dt
            Phi += cur_rot.dot(cur_n_jac_v_ba) * cur_n_dt * n_nn_dt
            Gamma = nnext_pos * cur_n_dt + cur_pos * n_nn_dt + cur_rot.dot(cur_n_delta_pos) * n_nn_dt - next_pos * (
                cur_n_dt + n_nn_dt) - next_rot.dot(n_nn_delta_pos) * cur_n_dt + cur_rot.dot(
                cur_n_delta_speed) * cur_n_dt * n_nn_dt

            A[3 * i: 3 * (i + 1), :] = Phi
            B[3 * i: 3 * (i + 1), 0] = Gamma - Lambda

        u, s, v = np.linalg.svd(A, full_matrices=False)
        c = np.dot(u.T, B)
        w = np.linalg.solve(np.diag(s), c)
        self.ba = np.dot(v.T, w).ravel()
        print self.ba

    def est_gravity(self):
        C = np.zeros((3 * (self.frame_size - 2), 3))
        D = np.zeros((3 * (self.frame_size - 2), 1))
        for i in range(self.frame_size - 2):
            cur_frame = self.imu_preint_array[i]
            next_frame = self.imu_preint_array[i + 1]
            nnext_pose = self.frames[i + 2]
            cur_n_dt = cur_frame.delta_time
            n_nn_dt = next_frame.delta_time

            cur_rot = cur_frame.init_pose[:3, :3]
            next_rot = next_frame.init_pose[:3, :3]
            cur_pos = cur_frame.init_pose[:3, 3]
            next_pos = next_frame.init_pose[:3, 3]
            nnext_pos = nnext_pose[1:4]

            cur_n_delta_pos = cur_frame.delta_pos
            n_nn_delta_pos = next_frame.delta_pos
            cur_n_delta_speed = cur_frame.delta_speed
            cur_n_jac_p_ba = cur_frame.jac_p_bias_a
            cur_n_jac_v_ba = cur_frame.jac_v_bias_a
            n_nn_jac_p_ba = next_frame.jac_p_bias_a

            Lambda = 0.5 * (cur_n_dt ** 2 * n_nn_dt + cur_n_dt * n_nn_dt * n_nn_dt) * np.eye(3)
            Phi = next_rot.dot(n_nn_jac_p_ba).dot(settings.bias_acc_prior) * cur_n_dt
            Phi += cur_rot.dot(cur_n_jac_p_ba).dot(settings.bias_acc_prior) * n_nn_dt
            Phi += cur_rot.dot(cur_n_jac_v_ba).dot(settings.bias_acc_prior) * cur_n_dt * n_nn_dt
            Gamma = nnext_pos * cur_n_dt + cur_pos * n_nn_dt + cur_rot.dot(cur_n_delta_pos) * n_nn_dt - next_pos * (
                cur_n_dt + n_nn_dt) - next_rot.dot(n_nn_delta_pos) * cur_n_dt + cur_rot.dot(
                cur_n_delta_speed) * cur_n_dt * n_nn_dt

            C[3 * i: 3 * (i + 1), :] = Lambda
            D[3 * i: 3 * (i + 1), 0] = Gamma - Phi

        u, s, v = np.linalg.svd(C, full_matrices=False)
        c = np.dot(u.T, D)
        w = np.linalg.solve(np.diag(s), c)
        x = np.dot(v.T, w)
        g0 = x / np.linalg.norm(x) * settings.Gravity
        self.gravity = g0.ravel()
        print g0
