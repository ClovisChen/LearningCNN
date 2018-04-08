#!/usr/bin/env python
# -*- coding: utf-8 -*-

import SE3
import SO3
import numpy as np
import sophus
import time


def test_se3_exp_map():
    print "testing SE3 exp map"
    _se3_ori = np.random.random(6) / 10

    print "the origin SE3 data is "
    print _se3_ori
    tic0 = time.clock()
    _SE3 = SE3.exp_se3(_se3_ori)
    toc0 = time.clock()
    print "time cost of package ", toc0 - tic0
    tic1 = time.clock()
    _SE3_ = sophus.SE3.exp(_se3_ori)
    toc1 = time.clock()
    print "time cost of sophus ", toc1 - tic1
    print _SE3
    print _SE3_


def test_se3_log_map():
    print "testing SE3 log map"
    _se3_ori = np.random.random(6) / 10
    _SE3 = sophus.SE3.exp(_se3_ori).matrix()
    print "the origin SE3 data is "
    print _SE3
    tic0 = time.clock()
    _se3 = SE3.log_se3(_SE3)
    toc0 = time.clock()
    print "time cost of package ", toc0 - tic0
    tic1 = time.clock()
    _SE3_ = sophus.SE3(_SE3)
    _se3_ = sophus.SE3.log(_SE3_)
    toc1 = time.clock()
    print "time cost of sophus ", toc1 - tic1
    print _se3, type(_se3)
    print _se3_.flatten(), type(_se3_)


def test_so3_log_map():
    _so3_ori = np.random.random(3) / 10
    _SO3 = sophus.SO3.exp(_so3_ori).matrix()
    print "the origin SO3 data is "
    print _SO3
    tic0 = time.clock()
    _so3 = SO3.log_so3(_SO3)
    toc0 = time.clock()
    print "time cost of package ", toc0 - tic0
    tic1 = time.clock()
    _so3_ = sophus.SO3.log(sophus.SO3(_SO3))
    toc1 = time.clock()
    print "time cost of sophus ", toc1 - tic1
    print _so3
    print _so3_.flatten()


def test_so3_exp_map():
    _so3_ori = np.random.random(3) / 10
    print "the origin so3 data is "
    print _so3_ori
    tic0 = time.clock()
    _SO3 = SO3.exp_so3(_so3_ori)
    toc0 = time.clock()
    print "time cost of package ", toc0 - tic0
    tic1 = time.clock()
    _SO3_ = sophus.SO3.exp(_so3_ori).matrix()
    toc1 = time.clock()
    print "time cost of sophus ", toc1 - tic1
    print _SO3
    print _SO3_

if __name__ == '__main__':
    test_se3_log_map()
    test_se3_exp_map()
    test_so3_log_map()
    test_so3_exp_map()