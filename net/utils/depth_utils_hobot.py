#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import contextlib

sys.path.insert(0, '.')
import net.monodepth_main
import pose.Frame
import pose.track
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import data.kitti_raw_loader
import time
import prepare_data
import yaml
from collections import namedtuple
import utils.transform
import data.pose_utils
from utils import *


def test_intrinsic():
    data_path = '/home/bobin/data/kitti/raw/'

    calib_file = data_path + '2011_09_26/calib_cam_to_cam.txt'

    dataloader = data.kitti_raw_loader.kitti_raw_loader()
    filedata = dataloader.read_raw_calib_file(calib_file)
    P_rect_02 = np.reshape(filedata['P_rect_' + '02'], (3, 4))
    P_rect_03 = np.reshape(filedata['P_rect_' + '03'], (3, 4))
    intrinsics_02 = P_rect_02[:3, :3]
    intrinsics_03 = P_rect_03[:3, :3]

    with printoptions(precision=3, suppress=True):
        print intrinsics_02
        print intrinsics_03
        print P_rect_02[:3, 3]
        print P_rect_03[:3, 3]

        print filedata['R_02']
        print filedata['R_03']
        print filedata['T_02']
        print filedata['T_03']




def test_mono_depth():
    params = net.monodepth_main.monodepth_parameters(
        encoder='vgg',
        height=256,
        width=512,
        batch_size=8,
        num_threads=8,
        num_epochs=50,
        do_stereo=False,
        wrap_mode='border',
        use_deconv=False,
        alpha_image_loss=0.85,
        disp_gradient_loss_weight=0.1,
        lr_loss_weight=1.0,
        full_summary=True)

    root_path = '/home/users/wendong.ding/code/sfm/learningReloc/'
    data_root = '/home/users/wendong.ding/data/hobot/'
    test_params = test_parameters(
        root_path=root_path,
        data_path=data_root + 'road/images/',
        filenames_file=root_path + 'net/utils/filenames/hobot_files.txt',
        dataset='hobot',
        mode='test',
        checkpoint_path=root_path + 'net/data/model/model_kitti',
        log_directory=root_path + 'net/data/log/',
        output_directory=root_path + 'net/data/output/',
        calib_int_file=data_root + 'camera/ov580/intrinsics.yml',
        calib_ext_file=data_root + '/camera/ov580/extrinsics.yml',
        trajectory_file=data_root + 'road/tra.txt',
        ground_truth_image=data_root + 'road/gt_image.npy'
    )

    with open(test_params.calib_int_file, 'r') as f:
        int_param = yaml.load(f)
        K_l = int_param["M1"]
        K_r = int_param["M2"]
        D_l = int_param["D1"]
        D_r = int_param["D2"]

    with open(test_params.calib_ext_file, 'r') as f:
        ext_param = yaml.load(f)
        P_l = ext_param["P1"]
        P_r = ext_param["P2"]
        R_l = ext_param["R1"]
        R_r = ext_param["R2"]
        R = ext_param["R"]
        T = ext_param["T"]

    bf = T[2]
    bf *= K_l[0, 0]
    dataloader = net.monodepth_main.MonodepthDataloader(test_params.data_path, test_params.filenames_file, params,
                                                        test_params.dataset, test_params.mode)

    trajectory = prepare_data.load_velodyne_trajectory(test_params.trajectory_file)
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch

    model = net.monodepth_main.MonodepthModel(params, test_params.mode, left, right)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = test_params.checkpoint_path
    train_saver.restore(sess, restore_path)

    num_test_samples = net.monodepth_main.count_text_lines(test_params.filenames_file)

    print('now testing {} files'.format(num_test_samples))

    disparities = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    mark = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    gt_idx = np.load(test_params.ground_truth_image)
    pcd = None
    #    for step in range(num_test_samples):

    for step in range(200):
        # tic = time.clock()
        disp, left_image = sess.run([model.disp_left_est[0], model.left])
        # left_image = sess.run(model.left)

        # toc = time.clock()
        # print 'time cost', toc - tic, 'in step ', step

        idepth = disp[0].squeeze()
        disparities[step] = idepth
        disparities_pp[step] = net.monodepth_main.post_process_disparity(disp.squeeze())

        im = idepth
        # show_depth(left_image[0], 0)
        # cv2.imwrite(test_params.output_directory + 'depth_%.6d.png'%step, im)
        depth = bf / idepth
        img0 = np.uint8(left_image[0] * 255)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        pose_vec = trajectory[gt_idx[step * 10]]
        Bias = utils.transform.rotation_matrix(-0.5 * np.pi, [0, 0, 1])
        Bias *= utils.transform.rotation_matrix(-0.5 * np.pi, [1, 0, 0])
        trans = pose_vec[1:4]
        x, y, z, w = pose_vec[4:]
        quat = [w, x, y, z]
        rot = data.pose_utils.quat2mat(quat)
        pose_mat = np.zeros([4, 4])
        pose_mat[:3, :3] = rot.dot(Bias[:3, :3])
        pose_mat[:3, 3] = trans
        # pose_mat = Bias.dot(pose_mat)
        print trans
        points = triangulate(img, depth, pose_mat, K_l)
        if pcd is None:
            pcd = points
        else:
            pcd = np.concatenate((pcd, points), axis=0)

    print('writing disparities.')
    # output_directory = output_directory
    np.save(test_params.output_directory + '/disparities.npy', disparities)
    # np.save(test_params.output_directory + '/disparities_pp.npy', disparities_pp)
    np.save(test_params.output_directory + '/hobot/map.npy', pcd)
    print('done.')


def show_npy(data_set):
    data_path = '/home/bobin/code/net/geometric/learningReloc/net/data/model/output/'
    disp_file_name = 'disparities_pp.npy'
    disp_pp_file_name = 'disparities_pp.npy'
    depth = np.load(data_path + data_set + disp_file_name)
    # depth_pp = np.load(data_path + disp_pp_file_name)
    # plt.ion()
    for i in range(depth.shape[0]):
        # img = np.append(depth[i], depth_pp[i], axis=1)
        img = depth[i]
        # plt.figure(0)
        # plt.imshow(ori)
        # plt.savefig(data_path + 'gray%.6d.png' % (i + 1), bbox_inches='tight')
        # plt.figure(1)
        plt.imshow(img)
        # plt.pause(0.001)
        plt.savefig(data_path + data_set + 'disparity%.6d.png' % (i + 1), bbox_inches='tight')


if __name__ == '__main__':
    # show_npy('hobot_garage/')
    test_mono_depth()
