#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
import tensorflow as tf
import logging
import scipy as scp
from collections import namedtuple
import cv2
import tensorvision.utils as tv_utils
import tensorvision.core as tv_core
import seg_utils.seg_utils as tv_seg
import time
import net.monodepth_main
import net.monodepth_dataloader
import net.utils.utils
import model.architecture as arch
import model.objective as objective

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

file_parameters = namedtuple('parameters',
                             'root_path, '
                             'data_path, '
                             'log_directory, '
                             'runs_dir')

class seg_net():
    def __init__(self):
        self.sess = None
        self.model = None
        flags = tf.app.flags
        self.FLAGS = flags.FLAGS

    def build_net(self, file_params):
        root_path = file_params.root_path
        logdir = file_params.log_directory
        hypes = tv_utils.load_hypes_from_logdir(root_path, json_file='dhypes.json')
        self.image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(self.image_pl, 0)
        logits = arch.inference(hypes, image, train=False)
        prediction = objective.decoder(hypes, logits, train=False)
        self.sess = tf.Session()
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coordinator)
        tv_core.load_weights(logdir, self.sess, saver)
        self.softmax = prediction['softmax']

    def run_sess(self, image):
        feed = {self.image_pl: image}
        output = self.sess.run([self.softmax], feed_dict=feed)
        return output

    def load_data_tf(self, net_params, test_params):
        dataloader = net.monodepth_dataloader.MonodepthDataloader(test_params.data_path, test_params.filenames_file,
                                                                  net_params,
                                                                  test_params.dataset, test_params.mode)

        left = dataloader.left_image_batch
        num_test_samples = net.monodepth_main.count_text_lines(test_params.filenames_file)
        return num_test_samples, left

    def load_data(self, file_names):
        with open(file_names) as fp:
            data = fp.read()
            lines = data.split('\n')
            lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
                     len(line) > 0 and line[0] != "#"]
            return lists


def test_park_iacas():
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
    data_path = '/home/chen-tian/data/SelfData/apple/'
    runs_dir = 'RUNS/KittiSeg_pretrained/'
    net_dir = 'homo_net/'
    test_params = net.utils.utils.test_parameters(
        root_path=root_path + net_dir,
        data_path=data_path,
        filenames_file=root_path + 'net/utils/filenames//kitti_odom_color_depth.txt',
        dataset='kitti',
        mode='test',
        checkpoint_path=root_path + 'net/data/model/model_kitti',
        log_directory=root_path + net_dir + runs_dir,
        output_directory=data_path + 'learningReloc/output/',
        kitti_calib=data_path + 'dataset/sequences/00/calib.txt',
        trajectory_file=data_path + 'dataset/poses/00.txt',
        height_origin=370,
        width_origin=1226,
        calib_ext_file='',
        calib_int_file='',
        ground_truth_image=''
    )
    video_name = '/home/chen-tian/data/SelfData/apple/IMG_0015.MOV'
    cap = cv2.VideoCapture(video_name)

    segnet = seg_net()
    segnet.build_net(test_params)
    image_lists = segnet.load_data(test_params.filenames_file)
    cnt = 0
    ret, frame = cap.read()
    while ret:
        #cv2.imshow('test', frame)
        ret, frame = cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        img = cv2.resize(img, (w / 4, h / 4))
        # for left, right in image_lists:
        # img = cv2.imread(test_params.data_path + left, cv2.IMREAD_UNCHANGED)
        # img = scp.misc.imread(test_params.data_path + left)
        tic = time.clock()
        output = segnet.run_sess(img)
        toc = time.clock()
        print 'time cost', toc - tic
        shape = img.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        rb_image = tv_seg.make_overlay(img, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a 'hard' prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold
        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(img, street_prediction)

        cv2.imshow('kitti', green_image)
        cv2.imwrite(test_params.output_directory + '/%d.png'%cnt, green_image)
        cnt += 1
        cv2.waitKey(10)

        # cap.open(video_name)


def test_kitti_odometry():
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
    data_path = '/home/chen-tian/data/KITTI/odom/'
    runs_dir = 'RUNS/KittiSeg_pretrained/'
    net_dir = 'homo_net/'
    test_params = net.utils.utils.test_parameters(
        root_path=root_path + net_dir,
        data_path=data_path,
        filenames_file=root_path + 'net/utils/filenames//kitti_odom_color_depth.txt',
        dataset='kitti',
        mode='test',
        checkpoint_path=root_path + 'net/data/model/model_kitti',
        log_directory=root_path + net_dir + runs_dir,
        output_directory=data_path + 'learningReloc/output/',
        kitti_calib=data_path + 'dataset/sequences/00/calib.txt',
        trajectory_file=data_path + 'dataset/poses/00.txt',
        height_origin=370,
        width_origin=1226,
        calib_ext_file='',
        calib_int_file='',
        ground_truth_image=''
    )

    segnet = seg_net()
    segnet.build_net(test_params)
    image_lists = segnet.load_data(test_params.filenames_file)
    cnt = 0
    for left, right in image_lists:
        # img = cv2.imread(test_params.data_path + left, cv2.IMREAD_UNCHANGED)
        img = scp.misc.imread(test_params.data_path + left)
        tic = time.clock()
        output = segnet.run_sess(img)
        toc = time.clock()
        print 'time cost', toc - tic
        shape = img.shape
        output_image = output[0][:, 1].reshape(shape[0], shape[1])

        # Plot confidences as red-blue overlay
        rb_image = tv_seg.make_overlay(img, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a 'hard' prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold
        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(img, street_prediction)

        cv2.imshow('kitti', output_image)
        cv2.imwrite(test_params.output_directory + '/segment/%.6d.png' % cnt, int(255*output_image))
        cnt += 1
        cv2.waitKey(10)


if __name__ == '__main__':
    test_kitti_odometry()
