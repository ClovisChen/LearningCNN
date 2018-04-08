#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import cv2
import numpy as np
import yaml
import tensorflow as tf
import sys
import glob

sys.path.insert(0, '.')
from net.monodepth_model import *
from net.monodepth_main import *
import matplotlib.pyplot as plt
from collections import namedtuple
import prepare_data
data_parameters = namedtuple('parameters',
                             'data_root, '
                             'trajectory, '
                             'left_path, right_path, '
                             'bag_name, '
                             'intrinsic_name, '
                             'extrinsic_name, '
                             'image_topic, '
                             'image_width, '
                             'image_height')


# A yaml constructor is for loading from a yaml node.
# This is taken from @misha 's answer: http://stackoverflow.com/a/15942429
def opencv_matrix_constructor(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    if mapping["cols"] > 1:
        mat.resize(mapping["rows"], mapping["cols"])
    else:
        mat.resize(mapping["rows"], )
    return mat


yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix_constructor)


# A yaml representer is for dumping structs into a yaml node.
# So for an opencv_matrix type (to be compatible with c++'s FileStorage) we save the rows, cols, type and flattened-data
def opencv_matrix_representer(dumper, mat):
    if mat.ndim > 1:
        mapping = {'rows': mat.shape[0], 'cols': mat.shape[1], 'dt': 'd', 'data': mat.reshape(-1).tolist()}
    else:
        mapping = {'rows': mat.shape[0], 'cols': 1, 'dt': 'd', 'data': mat.tolist()}
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)


yaml.add_representer(np.ndarray, opencv_matrix_representer)


def get_rect_parameters(width, height, int_fn, ext_fn):
    with open(int_fn, 'r') as f:
        int_param = yaml.load(f)
        K_l = int_param["M1"]
        K_r = int_param["M2"]
        D_l = int_param["D1"]
        D_r = int_param["D2"]

    with open(ext_fn, 'r') as f:
        ext_param = yaml.load(f)
        P_l = ext_param["P1"]
        P_r = ext_param["P2"]
        R_l = ext_param["R1"]
        R_r = ext_param["R2"]
        R = ext_param["R"]
        T = ext_param["T"]

    rows_l = height
    cols_l = width
    rows_r = rows_l
    cols_r = cols_l
    M1l, M1r = cv2.initUndistortRectifyMap(K_l, D_l, R_l, P_l[:3, :3], (cols_l, rows_l), cv2.CV_32F)
    M2l, M2r = cv2.initUndistortRectifyMap(K_r, D_r, R_r, P_r[:3, :3], (cols_r, rows_r), cv2.CV_32F)
    return M1l, M1r, M2l, M2r


def save_bag_images(params):
    M1l, M1r, M2l, M2r = get_rect_parameters(params.image_width, params.image_height, params.intrinsic_name,
                                             params.extrinsic_name)
    trajectory = prepare_data.load_velodyne_trajectory(params.trajectory)
    bag = rosbag.Bag(params.data_root + params.bag_name)
    file_count = 0
    gt_idx = int(0)
    gt_image = list()
    for topic, msg, t in bag.read_messages(topics=[params.image_topic]):
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        input_height = image_np.shape[0]
        input_width = image_np.shape[1] / 2
        right_im = image_np[:input_height, :input_width]
        left_im = image_np[:input_height, input_width:]
        # left_un_im = cv2.remap(left_im, M1l, M1r, cv2.INTER_LINEAR)
        # right_un_im = cv2.remap(right_im, M2l, M2r, cv2.INTER_LINEAR)
        left_file_name = params.data_root + params.left_path + "%.6d.png" % file_count
        right_file_name = params.data_root + params.right_path + "%.6d.png" % file_count
        # cv2.imwrite(left_file_name, left_im)
        # cv2.imwrite(right_file_name, right_im)
        # cv2.imshow('left_image', left_im)
        # print  t.to_sec()
        if gt_idx < len(trajectory):
            while trajectory[gt_idx, 0] < t.to_sec():
                gt_idx += 1
                if gt_idx >= len(trajectory):
                    break
        gt_image.append(gt_idx)
        print gt_idx
        # cv2.waitKey(1)
        file_count += 1
    np.save('gt_image.npy', gt_image)


def make_file_name_list():
    data_root = '/home/users/wendong.ding/data/hobot/0830/images/'
    left_file_names = glob.glob(data_root + 'left/*.png')
    right_file_names = glob.glob(data_root + 'right/*.png')
    with open('file_names.txt', 'w') as fp:
        for l, r in zip(left_file_names, right_file_names):
            fp.write(l)
            fp.write('\t')
            fp.write(r)
            fp.write('\n')


def extract_ros_msg():
    root_path = '/home/users/wendong.ding/code/sfm/learningReloc/'
    checkpoint_path = root_path + 'net/data/model/model_kitti'
    log_directory = root_path + 'net/data/log/'
    output_directory = root_path + 'net/data/output/'

    M1l, M1r, M2l, M2r = get_rect_parameters()
    params = monodepth_parameters(
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

    left = tf.placeholder(tf.float32, [2, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    # right = tf.placeholder(tf.float32, [2, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    model = MonodepthModel(params, "test", left, None)
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
    restore_path = checkpoint_path
    train_saver.restore(sess, restore_path)

    bag = rosbag.Bag(bag_filename)
    for topic, msg, t in bag.read_messages(topics=[image_topics]):
        np_arr = np.fromstring(msg.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        input_height = image_np.shape[0]
        input_width = image_np.shape[1] / 2
        right_im = image_np[:input_height, :input_width]
        left_im = image_np[:input_height, input_width:]

        left_un_im = cv2.remap(left_im, M1l, M1r, cv2.INTER_LINEAR)
        right_un_im = cv2.remap(right_im, M2l, M2r, cv2.INTER_LINEAR)
        show_im_left = cv2.resize(left_un_im, (input_width / 2, input_height / 2))
        show_im_right = cv2.resize(right_un_im, (input_width / 2, input_height / 2))
        left_image = np.tile((left_un_im, left_un_im, left_un_im), 2)
        right_image = np.tile((right_un_im, right_un_im, right_un_im), 2)
        left_image = left_image.astype(np.float32) / 255
        left_images = np.stack((left_image, np.fliplr(left_image)), 0)

        right_image = right_image.astype(np.float32) / 255
        right_images = np.stack((right_image, np.fliplr(right_image)), 0)

        disp = sess.run(model.disp_left_est[0], feed_dict={left: left_images})
        disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

        disparities = disp[0].squeeze()
        # output_directory = os.path.dirname(args.image_path)
        # output_name = os.path.splitext(os.path.basename(args.image_path))[0]
        plt.figure(0)
        plt.imshow(disparities)
        plt.pause(0.01)

        # cv2.imshow('left_image', show_im_left)
        # cv2.imshow('right_image', show_im_right)
        # cv2.waitKey(10)

    bag.close()


if __name__ == '__main__':
    ### data for hobot sh ground data
    params = data_parameters(
        data_root='/home/bobin/data/hobot/data-sh/sensor_20170803-154153_/',
        bag_name='20170803-154153.bag',
        left_path="hobot/left/",
        right_path="hobot/right/",
        intrinsic_name='/home/bobin/data/hobot/data-sh/sensor_20170803-154153_/calib/intrinsics.yml',
        extrinsic_name='/home/bobin/data/hobot/data-sh/sensor_20170803-154153_/calib/extrinsics.yml',
        image_topic='/sensor/stereo_camera_front/stereo_gray/compressed',
        image_width=1280,
        image_height=720,
        trajectory='/home/bobin/data/hobot/data-sh/sensor_20170803-154153_/tra.txt'
    )

    ## data for hobot hailong garage
    # params = data_parameters(
    #     data_root='/home/bobin/data/hobot/0830/',
    #     bag_name='20170830-180741.bag',
    #     left_path="image/left/",
    #     right_path="image/right/",
    #     intrinsic_name='/home/bobin/data/hobot/0830/calib/intrinsics.yml',
    #     extrinsic_name='/home/bobin/data/hobot/0830/calib/extrinsics.yml',
    #     image_topic='/sensor/uvc_sensor/image_all/compressed',
    #     image_width=1280,
    #     image_height=720
    # )

    save_bag_images(params)
    # make_file_name_list()
