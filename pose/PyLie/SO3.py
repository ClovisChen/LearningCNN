#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import Quaternion as quat
import eulerangles as euler
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


def exp_so3(so3):
    """
    :param so3:
    :return:
    """
    phi = so3
    theta = np.linalg.norm(phi)
    if theta == 0:
        SO3 = np.eye(3)
    else:
        alpha = phi / theta
        alpha = np.nan_to_num(alpha)
        alpha = np.expand_dims(alpha, axis=0)  # row vector
        SO3 = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * alpha.T.dot(alpha) + np.sin(theta) * skew_matrix(alpha)
    return SO3


def log_so3(SO3):
    rot = SO3
    theta = np.arccos((np.trace(rot) - 1) / 2)
    if theta == 0:
        alpha = np.zeros(3)
    else:
        lnR = theta / 2 / np.sin(theta) * (rot - rot.T)
        alpha = np.array([-lnR[1, 2], lnR[0, 2], -lnR[0, 1]])
    so3 = alpha
    return so3


class SO3():
    def __init__(self):
        self.rotation = np.eye(3)

    def __add__(self, other):
        self.rotation.dot(other.rotation)

    def __sub__(self, other):
        self.rotation.dot(np.linalg.inv(other.rotation))

    def log(self):
        return log_so3(self.rotation)

    def so3(self):
        return self.log()

    def exp(self):
        return self.rotation

    def rotation_matrix(self):
        return self.exp()

    def set_euler(self, _euler):
        x, y, z = _euler
        self.rotation = euler.euler2mat(z, x, y)

    def set_quaternion(self, _quaternion):
        self.rotation = quat.quat2mat(_quaternion)

    def set_axis_angle(self, theta, axis):
        mat = tf.rotation_matrix(theta, axis)
        self.rotation = mat[:3, :3]

    @staticmethod
    def right_jac(w):
        right_jac = np.eye(3)
        theta = np.linalg.norm(w)
        if theta < np.finfo(float).eps:
            return right_jac
        else:
            k = w / theta
            k_hat = skew_matrix(k)
            right_jac = right_jac - (1 - np.cos(theta)) / theta * k_hat + (1 - np.sin(theta) / theta) * k_hat.dot(k_hat)
            return right_jac

    @staticmethod
    def right_jac_inv(w):
        right_jac_inv = np.eye(3)
        theta = np.linalg.norm(w)
        if theta < np.finfo(float).eps:
            return right_jac_inv
        else:
            k = w / theta
            k_hat = skew_matrix(k)
            right_jac_inv += 0.5 * k_hat + (1.0 - (1.0 + np.cos(theta)) * theta / (2.0 * np.sin(theta))) * k_hat.dot(
                k_hat)
            return right_jac_inv

    @staticmethod
    def left_jac(w):
        return SO3.right_jac(-w)

    @staticmethod
    def left_jac_inv(w):
        return SO3.right_jac_inv(-w)
