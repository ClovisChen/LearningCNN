#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SE3
import SO3
import numpy as np
import sophus


def test_se3():
    _SE3 = SE3.SE3()
    _SE3.set_translation([10, 2, 30])
    # _SE3.set_translation(np.array([1e-10, 2, 3e-4]))
    # _SE3.set_translation([1e-10, 2, 3e-4])
    _SE3.set_euler([0.1, 0.2, 0.3])
    print _SE3.log()

    _SE3.set_axis_angle(np.pi / 3, [1, 0, 0])


    print _SE3.translation
    print _SE3.rotation_matrix()
    print _SE3.log()
    print _SE3.exp()

    _SE3_ = sophus.SE3()
    x, y, z = _SE3.translation
    _SE3_ *=_SE3_.trans(x, y, z)
    _SE3_ *= _SE3_.rotX(np.pi / 3)
    print _SE3_.translation()
    print _SE3_.rotationMatrix()

    print _SE3_.log().flatten()
    print _SE3.log()
    print SE3.exp_se3(_SE3_.log().flatten())
    print _SE3_.matrix()


if __name__ == '__main__':
    test_se3()
