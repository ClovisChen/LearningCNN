#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import contextlib

sys.path.insert(0, '.')
import net.monodepth_main
# import pose.Frame
# import pose.track
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import data.kitti_raw_loader
import time
import net.utils.utils
import prepare_data
import pcl


class Mapping():
    def __init__(self):
        self.calib_dataloader = None
        self.intrinsic = None
        self.bf = None
        self.model = None
        self.sess = None

    def calib_params(self, calib_file):
        self.calib_dataloader = data.kitti_raw_loader.kitti_raw_loader()
        filedata = self.calib_dataloader.read_raw_calib_file(calib_file)
        P_2 = np.reshape(filedata['P2'], (3, 4))
        P_3 = np.reshape(filedata['P3'], (3, 4))
        self.intrinsic = P_2[:3, :3]
        self.bf = P_2[0, 3] - P_3[0, 3]

    def load_trajectory(self, filename):
        self.trajectory = prepare_data.load_kitti_trajectory(filename)


    def build_net(self, net_params, test_params):
        dataloader = net.monodepth_main.MonodepthDataloader(test_params.data_path, test_params.filenames_file,
                                                            net_params,
                                                            test_params.dataset, test_params.mode)

        left = dataloader.left_image_batch
        right = dataloader.right_image_batch
        self.model = net.monodepth_main.MonodepthModel(net_params, test_params.mode, left, right)

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        # SAVER
        train_saver = tf.train.Saver()

        # INIT
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)

        # RESTORE
        restore_path = test_params.checkpoint_path
        train_saver.restore(self.sess, restore_path)

        num_test_samples = net.monodepth_main.count_text_lines(test_params.filenames_file)
        return num_test_samples

    def sess_run(self):
        disp, left_image = self.sess.run([self.model.disp_left_est[0], self.model.left])
        return disp, left_image


if __name__ == '__main__':
    mapping = Mapping()
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
    root_path = '/home/chen-tian/data/code/learningReloc/'
    data_path = '/home/chen-tian/data/SelfData/apple/exp2/'
    test_params = net.utils.utils.test_parameters(
        root_path=root_path,
        data_path=data_path,
        filenames_file=root_path + 'net/utils/filenames/file_apple_park_2.txt',
        dataset='kitti',
        mode='test',
        checkpoint_path=root_path + 'net/data/model/model_kitti',
        log_directory=data_path + 'learningReloc/log/',
        output_directory=data_path + 'learningReloc/output/depth/',
        kitti_calib=data_path + 'dataset/sequences/00/calib.txt',
        trajectory_file=data_path + 'dataset/poses/00.txt',
        height_origin=370,
        width_origin=1226,
        calib_ext_file='',
        calib_int_file='',
        ground_truth_image=''
    )
    num_test_samples = mapping.build_net(params, test_params)
    # mapping.load_trajectory(test_params.trajectory_file)
    # mapping.calib_params(test_params.kitti_calib)
    # assert num_test_samples == len(mapping.trajectory)
    print('now testing {} files'.format(num_test_samples))
    disparities_vector = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp_vector = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    # cloud = pcl.PointCloud()
    # points = None
    for step in range(num_test_samples):
        print step
        disp, left_image = mapping.sess_run()
        disparities = disp[0].squeeze()
        plt.imshow(disparities)
        plt.savefig(test_params.output_directory + '%.6d.png'%step)
        plt.pause(0.001)
        left_ori = np.uint8(left_image[0] * 255)
        cv2.imshow('left', left_ori)
        cv2.waitKey(10)
        # depth = mapping.bf / disparities
        # color_origin = np.uint8(left_image[0] * 255)
        # gray_orgin = cv2.cvtColor(color_origin, cv2.COLOR_BGR2GRAY)
        # pose_mat = mapping.trajectory[step].reshape((3, 4))
        # pcd = net.utils.utils.triangulate(gray_orgin, depth, pose_mat, mapping.intrinsic,
        #                                   (test_params.width_origin, test_params.height_origin))
        # if points is None:
        #     points = pcd
        # else:
        #     points = np.concatenate((points, pcd), axis=0)
    # cloud = pcl.PointCloud(np.float32(points))
    # cloud.to_file(test_params.output_directory + 'map.pcd')
