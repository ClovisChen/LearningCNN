#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SO3
import numpy as np


def test_so3():
    _SO3 = SO3.SO3()
    _SO3.set_euler([0.1, 0.2, 0.3])
    print _SO3.log()

    _SO3.set_euler([0., 0., 0.])
    print _SO3.log()

    _SO3.set_euler([-0.1, -0.2, -0.3])
    print _SO3.log()

    _SO3.set_axis_angle(np.pi / 3, [1, 0, 0])
    print _SO3.log()

    _SO3.set_axis_angle(np.pi / 2, [1, 0, 0])
    print _SO3.log()

    _SO3.set_axis_angle(np.pi / 2, [1, 1, 0])
    print _SO3.log()

    _SO3.set_axis_angle(np.pi / 2, [1, 1, 1])
    print _SO3.log()

    _SO3.set_axis_angle(np.pi / 2, [0, 0, -0.1])
    print _SO3.log()

    _SO3.set_quaternion([-1, 0, 0, 0])
    print _SO3.log()

    _SO3.set_quaternion([1, 0, 0, 0])
    print _SO3.log()

    print 'test right jacobian'
    w = np.array([0, 0, 0.1])
    w0 = np.array([0, 0, -0.1])
    right = SO3.exp_so3(_SO3.right_jac(w).dot(w))
    print SO3.exp_so3(w0).dot(right)
    print SO3.exp_so3(w0 + w)
    w = np.array([0, 1, 2])
    print _SO3.right_jac(w)


if __name__ == '__main__':
    test_so3()
