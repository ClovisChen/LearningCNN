#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
import SO3
import eulerangles as euler
import Quaternion as quat
import transform as tf

def skew_matrix(v):
    """
    给定向量，输入其skew matrix(hat)
    :param v: v \in \mathbb{R}^3
    :return: skew matrix \in \mathbb{R}^{3\times3}
    """
    if len(v) == 4: v = v[:3] / v[3]
    skv = np.roll(np.roll(np.diag(v.flatten()), 1, 1), -1, 0)
    return skv - skv.T


def exp_se3(se3):
    """
    :param se3:
    :return:
    """
    phi = se3[3:]  # about orientation
    rho = se3[:3]  # about translation
    theta = np.linalg.norm(phi)
    if theta == 0:
        rot = np.eye(3)
        J = np.eye(3)
    else:
        # with np.errstate(invalid='ignore'):
        alpha = phi / theta
        alpha = np.nan_to_num(alpha)
        alpha = np.expand_dims(alpha, axis=0)  # row vector
        rot = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * alpha.T.dot(alpha) + np.sin(theta) * skew_matrix(alpha)
        J = np.sin(theta) / theta * np.eye(3) + (1 - np.sin(theta) / theta) * alpha.T.dot(alpha) + (1 - np.cos(
            theta)) / theta * skew_matrix(alpha)
        J = np.nan_to_num(J)
    trans = J.dot(rho)
    SE3 = np.eye(4)
    SE3[:3, :3] = rot
    SE3[:3, 3] = trans
    return SE3


def log_se3(SE3):
    rot = SE3[:3, :3]
    trans = SE3[:3, 3]
    theta = np.arccos((np.trace(rot) - 1) / 2)
    if theta == 0:
        J = np.eye(3)
        alpha = np.zeros(3)
    else:
        lnR = theta / 2 / np.sin(theta) * (rot - rot.T)
        alpha = np.array([-lnR[1, 2], lnR[0, 2], -lnR[0, 1]])
        J = np.sin(theta) / theta * np.eye(3) + (1 - np.sin(theta) / theta) * alpha.T.dot(alpha) + (1 - np.cos(
            theta)) / theta * skew_matrix(alpha)
    rho = np.linalg.inv(J).dot(trans)
    # rho = J.dot(trans)
    # K = scipy.linalg.cho_factor(J)
    # rho = scipy.linalg.cho_solve(K, trans)
    se3 = np.zeros(6)
    se3[:3] = rho
    se3[3:] = alpha
    return se3


class SE3():
    def __init__(self):
        self.translation = np.zeros(3)
        self.rotation = np.eye(3)

    def __add__(self, other):
        self.translation += other.translation()
        self.rotation.dot(other.rotation_matrix())

    def __sub__(self, other):
        self.translation -= other.translation()
        self.rotation.dot(np.linalg.inv(other.rotation_matrix()))

    def translation(self):
        return self.translation

    def so3(self):
        return SO3.log_so3(self.rotation)

    def rotation_matrix(self):
        return self.rotation

    def inverse(self):
        inv = np.eye(4)
        inv[:3, :3] = self.rotation.T
        inv[:3, 3] = self.translation
        return inv

    def se3(self):
        _se3 = log_se3(self.exp())
        return _se3

    def log(self):
        return self.se3()

    def exp(self):
        _SE3 = np.eye(4)
        _SE3[:3, :3] = self.rotation
        _SE3[:3, 3] = self.translation
        return _SE3

    def set_translation(self, _translation):
        self.translation = _translation

    def set_euler(self, _euler):
        x, y, z = _euler
        self.rotation = euler.euler2mat(z, x, y)

    def set_quaternion(self, _quaternion):
        self.rotation = quat.quat2mat(_quaternion)

    def set_axis_angle(self, theta, axis):
        mat = tf.rotation_matrix(theta, axis)
        self.rotation = mat[:3, :3]