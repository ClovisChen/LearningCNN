#!/usr/bin/env python
# -*- coding: utf-8 -*-are not covered by the UCLB ACP-A Licence,

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from fusion_net.base_fn import *
from fusion_net.bilinear_sampler import *

monodepth_parameters = namedtuple('parameters',
                        'encoder, '
                        'height, width, '
                        'batch_size, '
                        'num_threads, '
                        'num_epochs, '
                        'do_stereo, '
                        'wrap_mode, '
                        'use_deconv, '
                        'alpha_image_loss, '
                        'disp_gradient_loss_weight, '
                        'lr_loss_weight, '
                        'full_summary, '
                        'seg_loss')

class Model(object):
    def __init__(self, params, mode, left, right, labels, reuse_variables=None, model_index=0):
        self.params = params
        self.mode = mode
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()
        self.labels=labels

    def build_vgg(self):
        # set convenience functions
        # conv = self.conv
        if self.params.use_deconv:
            # upconv = self.deconv
            upconv = deconv
        # else:
        #     upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = conv_block(self.model_input, 32, 7)  # H/2
            conv2 = conv_block(conv1, 64, 5)  # H/4
            conv3 = conv_block(conv2, 128, 3)  # H/8
            conv4 = conv_block(conv3, 256, 3)  # H/16
            conv5 = conv_block(conv4, 512, 3)  # H/32
            conv6 = conv_block(conv5, 512, 3)  # H/64
            conv7 = conv_block(conv6, 512, 3)  # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6

        with tf.variable_scope('depth_decoder'):
            upconv7 = upconv(conv7, 512, 3, 2)  # H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7 = conv(concat7, 512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2)  # H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6 = conv(concat6, 512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2)  # H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5 = conv(concat5, 256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2)  # H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4 = conv(concat4, 128, 3, 1)
            self.disp4 = get_disp(iconv4)
            udisp4 = upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4, 64, 3, 2)  # H/4
            concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            iconv3 = conv(concat3, 64, 3, 1)
            self.disp3 = get_disp(iconv3)
            udisp3 = upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3, 32, 3, 2)  # H/2
            concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            iconv2 = conv(concat2, 32, 3, 1)
            self.disp2 = get_disp(iconv2)
            udisp2 = upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2, 16, 3, 2)  # H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1 = conv(concat1, 16, 3, 1)
            self.disp1 = get_disp(iconv1)

    # ---------------------------seg------------------------------------------
        with tf.variable_scope('seg_decoder'):
            fcn_in = conv7
            num_classes = 2
            fcn_in = tf.Print(fcn_in, [tf.shape(fcn_in)], message='shape of %s' % fcn_in.name, summarize=4, first_n=1)
            scale_down = 1
            he_init = tf.contrib.layers.variance_scaling_initializer()
            l2_regularizer = tf.contrib.layers.l2_regularizer(0.0005)

            # score layer
            score_fr = tf.layers.conv2d(fcn_in, kernel_size=[1, 1], filters=num_classes, padding='SAME',
                                        kernel_initializer=he_init, kernel_regularizer=l2_regularizer)
            activation_summary(score_fr)

            # do first upsamling
            upscore2 = upscore_layer(
                score_fr, upshape=tf.shape(skip6), num_classes=num_classes, name='upscore2', ksize=4, stride=2 )
            he_init2 = tf.contrib.layers.variance_scaling_initializer(factor=2.0*scale_down)

            #score feed2
            score_feed2 = tf.layers.conv2d(skip6, kernel_size=[1, 1], filters=num_classes, padding='SAME',
                                   name='score_feed2', kernel_initializer=he_init2, kernel_regularizer=l2_regularizer)
            activation_summary(score_feed2)
            skip = True
            if skip:
                # create skip connection
                fuse_feed2 = tf.add(upscore2, score_feed2)
            else:
                fuse_feed2 = upscore2
                fuse_feed2.set_shape(score_feed2.shape)

            # Do second upsampling
            upscore4 = upscore_layer(fuse_feed2, upshape=tf.shape(skip5), num_classes=num_classes, name='upscore4',
                                     ksize=4, stride=2)
            he_init4 = tf.contrib.layers.variance_scaling_initializer(factor=2.0*scale_down*scale_down)

            # score feed4
            score_feed4 = tf.layers.conv2d(skip5, kernel_size=[1, 1], filters=num_classes, padding='SAME',
                                   name='score_feed4', kernel_initializer=he_init4, kernel_regularizer=l2_regularizer)
            activation_summary(score_feed4)
            if skip:
                # create second skip connection
                fuse_pool3 = tf.add(upscore4, score_feed4)
            else:
                fuse_pool3 = upscore4
                fuse_pool3.set_shape(score_feed4.shape)

            # Do final upsampling
            self.upscore32 = upscore_layer(fuse_pool3, upshape=tf.shape(self.left), num_classes=num_classes,
                                      name='upscore32', ksize=16, stride=8)
            self.softmax=add_softmax(self.upscore32)

    def seg_loss(self):
        """Calculate the loss from the logits and the labels.

        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size].

        Returns:
          loss: Loss tensor of type float.
        """
        logits = self.upscore32
        with tf.name_scope('seg_loss'):
            logits = tf.reshape(logits, (-1, 2))
            shape = [logits.get_shape()[0], 2]
            # epsilon = tf.constant(value=hypes['solver']['epsilon'])
            epsilon = tf.constant(value=1e-05)
            # logits = logits + epsilon
            labels = tf.to_float(tf.reshape(self.labels, (-1, 2)))
            softmax = tf.nn.softmax(logits)
            reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
            weight_loss = tf.add_n(tf.get_collection(reg_loss_col), name='total_loss_seg')

            if self.params.seg_loss == 'xentropy':
                cross_entropy_mean = compute_cross_entropy_mean(self.labels, softmax)
            elif self.params.seg_loss == 'softF1':
                cross_entropy_mean = compute_f1(self.labels, softmax, epsilon)
            elif self.params.seg_loss == 'softIU':
                cross_entropy_mean = compute_soft_ui(self.labels, softmax, epsilon)

            enc_loss = tf.add_n(tf.get_collection('losses'), name='total_loss_seg')
            dec_loss = tf.add_n(tf.get_collection('dec_losses'), name='total_loss_seg')
            fc_loss = tf.add_n(tf.get_collection('fc_wlosses'), name='total_loss_seg')

            use_fc_wd = True
            if use_fc_wd:
                weight_loss = enc_loss + dec_loss
            else:
                weight_loss = enc_loss + dec_loss + fc_loss

            total_loss = cross_entropy_mean + weight_loss

            losses_seg = {}
            losses_seg['total_loss_seg'] = total_loss
            losses_seg['xentropy'] = cross_entropy_mean
            losses_seg['weight_loss'] = weight_loss

        return losses_seg

    # ---------------------------seg------------------------------------------

    def depth_losses(self):
        with tf.variable_scope('losses', reuse=self.reuse_variables):
            # IMAGE RECONSTRUCTION
            # L1
            # 1. 原图和重建的图之间的差异，用L1范数表示
            self.l1_left = [tf.abs( self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_left  = [tf.reduce_mean(l) for l in self.l1_left]
            self.l1_right = [tf.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
            self.l1_reconstruction_loss_right = [tf.reduce_mean(l) for l in self.l1_right]

            # SSIM
            # 2. 原图和重建的图的SSIM，并且左右图加权求和 SSIM是Structural Similarity的简写，意思是结构相似性。
            # C ^ l_{ap} =\frac{1}{N}\sum_{i, j}\alpha\frac{1 - SSIM(I_{i, j} ^ l,\tilde{I_{i, j} ^ l})}{2} + (1 -\alpha) | | I ^ l_{ij} -\tilde{I_{ij}} | |
            # SSIM(x, y) =\frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x ^ 2 +\mu_y ^ 2 + c_1)(\sigma_x ^ 2 +\sigma_y ^ 2 + c_2)}
            # 用平均池化计算x y的均值。 \mu_x就是Ex
            # 用平均池化计算x ^ 2 y ^ 2 的均值再减去x y均值的平方算出方差。就是D(x) = E(x ^ 2) - (Ex) ^ 2
            # 用平均池化计算协方差 \sigma_{xy} = E(xy) - E(x)E(y)
            # 接着算分子和分母，并且算出SSIM
            # 最后计算 \frac{1 - SSIM(x, y)}{2} ，并用tf.clip_by_value()将它截断在0.0, 1.0之间。
            # 这样当x y越相似这一项就越小。
            self.ssim_left = [SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
            self.ssim_loss_left  = [tf.reduce_mean(s) for s in self.ssim_left]
            self.ssim_right = [SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
            self.ssim_loss_right = [tf.reduce_mean(s) for s in self.ssim_right]

            # WEIGTHED SUM
            self.image_loss_right = [self.params.alpha_image_loss * self.ssim_loss_right[i] + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_right[i] for i in range(4)]
            self.image_loss_left  = [self.params.alpha_image_loss * self.ssim_loss_left[i]  + (1 - self.params.alpha_image_loss) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
            self.image_loss = tf.add_n(self.image_loss_left + self.image_loss_right)

            # DISPARITY SMOOTHNESS   3. 视差图平滑损失  要根据尺度除以一个值，不然会因为尺度不同造成不平衡
            self.disp_left_loss  = [tf.reduce_mean(tf.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
            self.disp_right_loss = [tf.reduce_mean(tf.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
            self.disp_gradient_loss = tf.add_n(self.disp_left_loss + self.disp_right_loss)

            # LR CONSISTENCY  4. 左右图一致性损失
            # 意思是在左视差图中有一个偏移量，根据左视差图的索引和偏移量可以得到对应点在右图的位置索引。
            # 然后跑到右视差图中，根据这个位置索引和右视差图中的偏移量就可以算出左视差图当前索引的偏移量估计了。
            # 然后利用一个轮回回来的视差图偏移量估计和原来的偏移量相减并求出绝对值距离。这样所有点求和就可以求出一致性损失。
            self.lr_left_loss  = [tf.reduce_mean(tf.abs(self.right_to_left_disp[i] - self.disp_left_est[i]))  for i in range(4)]
            self.lr_right_loss = [tf.reduce_mean(tf.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in range(4)]
            self.lr_loss = tf.add_n(self.lr_left_loss + self.lr_right_loss)

            # TOTAL LOSS
            self.total_loss_depth = self.image_loss + self.params.disp_gradient_loss_weight * self.disp_gradient_loss + self.params.lr_loss_weight * self.lr_loss

    def build_losses(self, hypes, decoded_logits, labels):
        seg_loss=self.seg_loss(hypes, decoded_logits, labels)
        return self.total_loss_depth + seg_loss

    def build_model(self):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], activation_fn=tf.nn.elu):
            with tf.variable_scope('model', reuse=self.reuse_variables):

                self.left_pyramid  = scale_pyramid(self.left,  4)
                if self.mode == 'train':
                    self.right_pyramid = scale_pyramid(self.right, 4)

                if self.params.do_stereo:
                    self.model_input = tf.concat([self.left, self.right], 3)
                else:
                    self.model_input = self.left

                #build model
                # if self.params.encoder == 'vgg':
                self.build_vgg()
                # elif self.params.encoder == 'resnet50':
                #     self.build_resnet50()
                # else:
                #     return None

    def build_outputs(self):
        # STORE DISPARITIES  1. 视差图: 包括用来生成左图和生成右图的视差图
        #1. 将四个尺度的视差图排成队列
        #2. 从视差图队列中取出左(右)视差图(最后输出的视差图的0通道是左图,1通道是右图),然后给他们用tf.expand_dims()加上通道轴,变成[batch,height,width,1]形状的tensor.
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [tf.expand_dims(d[:,:,:,0], 3) for d in self.disp_est]
            self.disp_right_est = [tf.expand_dims(d[:,:,:,1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return

        # GENERATE IMAGES  2. 原图估计: 通过左(右)原图和右(左)图视差图生成右(左)图的估计
        with tf.variable_scope('images'):
            self.left_est  = [generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # LR CONSISTENCY  3. 一致性: 通过过右(左)视差图和左(右)视差图生成新的右(左)视差图:计算用来计算左右图一致性
        #用右视差图中的视差通过视差索引找到左视差图上的点,然后再通过做视差图点上的视差索引生成新的右视差图.就可以用右视差图和新的右视差图产生衡量一致性的项.
        #可以看到生成左右图的估计传入的是图和视差图.调用的是一维双线性采样函数.生成左图的时候视差图需要变成相反数输入.
        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS  4. 平滑性: 通过左(右)原图和左(右)图估计计算平滑项
        with tf.variable_scope('smoothness'):
            '''
            生成平滑项
            '''
            self.disp_left_smoothness  = get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
                tf.summary.scalar('lr_loss_' + str(i), self.lr_left_loss[i] + self.lr_right_loss[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)

                if self.params.full_summary:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if self.params.full_summary:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)

