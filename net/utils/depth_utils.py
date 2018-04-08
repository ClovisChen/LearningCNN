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


def show_depth(img, num_figure):
    plt.figure(num_figure)
    plt.imshow(img)
    plt.pause(0.01)
    # val_norm = np.linalg.norm(img)
    # img *= 255 / val_norm
    # val_max = np.max(img)
    # val_min = np.min(img)
    # val_scale = val_max - val_min
    # img = (img - val_min) / val_scale * 255
    # cimage = cv2.applyColorMap(img, cv2.COLORMAP_AUTUMN)
    # cv2.imshow(name, cimage)
    # cv2.waitKey(30)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


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
    data_path = '/home/users/wendong.ding/data/kitti/raw/'
    filenames_file = root_path + 'net/utils/filenames/kitti_test_files_png_2.txt'
    dataset = 'kitti'
    mode = 'test'
    checkpoint_path = root_path + 'net/data/model/model_kitti'
    log_directory = root_path + 'net/data/log/'
    output_directory = root_path + 'net/data/output/kitti/'
    calib_file = data_path + '2011_09_26/calib_cam_to_cam.txt'
    calib_dataloader = data.kitti_raw_loader.kitti_raw_loader()
    filedata = calib_dataloader.read_raw_calib_file(calib_file)
    P_rect_02 = np.reshape(filedata['P_rect_' + '02'], (3, 4))
    P_rect_03 = np.reshape(filedata['P_rect_' + '03'], (3, 4))
    intrinsics_02 = P_rect_02[:3, :3]
    intrinsics_03 = P_rect_03[:3, :3]
    bf = P_rect_02[0, 3] - P_rect_03[0, 3]
    bf /= intrinsics_02[0, 0]
    dataloader = net.monodepth_main.MonodepthDataloader(data_path, filenames_file, params, dataset, mode)
    left = dataloader.left_image_batch
    right = dataloader.right_image_batch
    model = net.monodepth_main.MonodepthModel(params, mode, left, right)

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
    if checkpoint_path == '':
        restore_path = tf.train.latest_checkpoint(log_directory + '/' + args.model_name)
    else:
        restore_path = checkpoint_path
    train_saver.restore(sess, restore_path)

    num_test_samples = net.monodepth_main.count_text_lines(filenames_file)
    d = [0.] * 5
    camera = pose.Frame.Camera(intrinsics_02[0, 0], intrinsics_02[1, 1], intrinsics_02[0, 2], intrinsics_02[1, 2],
                               512, 256, d, None)
    pyr_camera = pose.Frame.PyrCamera(camera, 4)
    tracker = pose.track.Tracker(camera)

    # print('now testing {} files'.format(num_test_samples))
    disparities_vector = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    disparities_pp_vector = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
    for step in range(num_test_samples):
        tic = time.clock()
        disp = sess.run(model.disp_left_est[0])
        left_image = sess.run(model.left)
        # print left_image.shape
        toc = time.clock()
        print 'time cost', toc - tic , "in " , step
       # imlr = np.append(left_image[0], left_image[1], axis=1)
        # imleft = left_image[0]
        disparities = disp[0].squeeze()
        disparities_vector[step] = disparities
        disparities_pp = net.monodepth_main.post_process_disparity(disp.squeeze())
        disparities_pp_vector[step] = disparities_pp
        #im = np.append(disparities, disparities_pp, axis=1)
        im = disparities
        #show_depth(im, 0)

        depth = bf / disparities
        img0 = np.uint8(left_image[0] * 255)
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        # print left_image.shape, left_image.dtype, np.max(left_image[0])
        frame = pose.Frame.Frame(step, img, None, depth, pyr_camera)
        # print left_image.shape
        markImage = tracker.lk_track(frame)
        #if markImage is not None:
           # show_depth(markImage, 1)


        ##cv2.imwrite(output_directory + '/test%d.jpg'%step, bf/im)
        # print bf/im
        # show_depth(disparities_pp, 1)
        # print('done.')
        # print('writing disparities.')
        # output_directory = output_directory
    np.save(output_directory + '/disparities.npy', disparities_vector)
    np.save(output_directory + '/disparities_pp.npy', disparities_pp_vector)
        # print('done.')


def show_npy(data_set):
    data_path = '/home/bobin/code/net/geometric/learningReloc/net/data/model/output/'
    disp_file_name = 'disparities.npy'
    disp_pp_file_name = 'disparities_pp.npy'
    depth = np.load(data_path + data_set + disp_file_name)
    depth_pp = np.load(data_path + data_set + disp_pp_file_name)
    # plt.ion()
    for i in range(depth.shape[0]):
        # file_name = data_path + "ori_i%.6d.png" % (i + 1)
        # ori = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # img = np.append(depth[i], depth_pp[i], axis=1)
        img = depth[i]
        plt.figure(0)
        # plt.imshow(ori)
        # plt.savefig(data_path + 'gray%.6d.png' % (i + 1), bbox_inches='tight')
        # plt.figure(1)
        plt.imshow(img)
        # plt.pause(0.001)
        plt.savefig(data_path + data_set + 'disparity%.6d.png' % (i + 1), bbox_inches='tight')


if __name__ == '__main__':
    # test_mono_depth()
    show_npy('kitti/')
    # show_npy('hobot/')
