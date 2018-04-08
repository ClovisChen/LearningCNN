#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
                        'full_summary')

def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def upsample_nn(x, ratio):
    s = tf.shape(x)
    h = s[1]
    w = s[2]
    return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

def scale_pyramid(img, num_scales):
    scaled_imgs = [img]
    s = tf.shape(img)
    h = s[1]
    w = s[2]
    for i in range(num_scales - 1):
        ratio = 2 ** (i + 1)
        nh = h // ratio
        nw = w // ratio
        scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
    return scaled_imgs

def generate_image_left(img, disp):
    return bilinear_sampler_1d_h(img, -disp)

def generate_image_right(img, disp):
    return bilinear_sampler_1d_h(img, disp)

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def get_disparity_smoothness(disp, pyramid):
    '''
    先计算视差图和原图在x和y上的梯度，然后计算x和y上的系数，最后用系数乘上视差图的梯度.
    x上的梯度是截取x轴的0到w-1的图，减去截取x轴的1到w的图. y同理.

    用数学式子表达： \Delta I=[I(b,y,x,c)-I(b,y,x+1,c), I(b,y,x,c)-I(b,y+1,x,c)]

    计算系数用数学式子表示： w=e^{-\frac{1}{|C|}\sum_c|g_{b,y,x,c}|} 注意g是原图的梯度。

    然后将系数和梯度对应相乘，把x与y的结果加在一起就是最后的平滑项。
    '''
    disp_gradients_x = [gradient_x(d) for d in disp]
    disp_gradients_y = [gradient_y(d) for d in disp]

    image_gradients_x = [gradient_x(img) for img in pyramid]
    image_gradients_y = [gradient_y(img) for img in pyramid]

    weights_x = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_x]
    weights_y = [tf.exp(-tf.reduce_mean(tf.abs(g), 3, keep_dims=True)) for g in image_gradients_y]

    smoothness_x = [disp_gradients_x[i] * weights_x[i] for i in range(4)]
    smoothness_y = [disp_gradients_y[i] * weights_y[i] for i in range(4)]
    return smoothness_x + smoothness_y

def get_disp(x):
    '''生成视差图
    在卷积之后送入sigmoid单元，这时结果就变成了在(0.0,1.0)的数。0.3是为了限制其大小的系数。这个值其实是相对于宽度width归一化的结果.
    记下来调用差值函数获得更好的图。最后把它变成[batch,height,width,channel]的形式输出.'''
    disp = 0.3 * conv(x, 2, 3, 1, tf.nn.sigmoid)
    #在upconv之后用了一个CNN加sigmoid函数，乘以0.3之后作为视差图
    return disp

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

def conv_block(x, num_out_layers, kernel_size):
    conv1 = conv(x,     num_out_layers, kernel_size, 1)
    conv2 = conv(conv1, num_out_layers, kernel_size, 2)
    return conv2

def maxpool(x, kernel_size):
    p = np.floor((kernel_size - 1) / 2).astype(np.int32)
    p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
    return slim.max_pool2d(p_x, kernel_size)

def upconv(x, num_out_layers, kernel_size, scale):
    #上卷积
    upsample = upsample_nn(x, scale)
    #先做最近邻上采样   最近邻上采样调用的是tensorflow.image里面的resize_nearest_neightbor函数
    up_conv = conv(upsample, num_out_layers, kernel_size, 1)
    #然后做卷积, 步长为1
    return up_conv

def deconv(x, num_out_layers, kernel_size, scale):
    p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
    conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
    return conv[:,3:-1,3:-1,:]

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

#---------------------------------seg-------------------------------------------

def add_softmax(logits):
    num_classes = 2
    with tf.name_scope('decoder'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-05)
        # logits = logits + epsilon

        softmax = tf.nn.softmax(logits)

    return softmax

def upscore_layer(bottom, upshape,
                   num_classes, name,
                   ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        in_features = bottom.get_shape()[3].value

        new_shape = [upshape[0], upshape[1], upshape[2], num_classes]
        output_shape = tf.stack(new_shape)

        f_shape = [ksize, ksize, num_classes, in_features]

        up_init = upsample_initilizer()

        weights = tf.get_variable(name="weights", initializer=up_init,
                                  shape=f_shape)

        tf.add_to_collection(tf.GraphKeys.WEIGHTS, weights)

        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

        deconv = tf.Print(deconv, [tf.shape(deconv)],
                          message='Shape of %s' % name,
                          summarize=4, first_n=1)

        _activation_summary(deconv)

    return deconv


def upsample_initilizer(self, dtype=dtypes.float32):
    """Returns an initializer that creates filter for bilinear upsampling.

    Use a transposed convolution layer with ksize = 2n and stride = n to
    perform upsampling by a factor of n.
    """
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-float point type.')

    def _initializer(shape, dtype=dtype, partition_info=None):
        """Initializer function."""
        if not dtype.is_floating:
            raise TypeError('Cannot create initializer for non-floating type.')

        width = shape[0]
        heigh = shape[0]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([shape[0], shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(shape)
        for i in range(shape[2]):
            '''
            the next line of code is correct as given
            [several issues were opened ...]
            we only want to scale each feature,
            so there is no interaction between channels,
            that is why only the diagonal i, i is initialized
            '''
            weights[:, :, i, i] = bilinear

        return weights

    return _initializer
# --------------------------Loss-------------------------------

def activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.

    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def compute_cross_entropy_mean(labels, softmax):
    # head = hypes['arch']['weight']
    head = [1, 1]
    cross_entropy = -tf.reduce_sum(tf.mul(labels * tf.log(softmax), head), reduction_indices=[1])

    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='xentropy_mean')
    return cross_entropy_mean

def compute_f1(labels, softmax, epsilon):
    labels = tf.to_float(tf.reshape(labels, (-1, 2)))[:, 1]
    logits = softmax[:, 1]
    true_positive = tf.reduce_sum(labels * logits)
    false_positive = tf.reduce_sum((1 - labels) * logits)

    recall = true_positive / tf.reduce_sum(labels)
    precision = true_positive / (true_positive + false_positive + epsilon)

    score = 2 * recall * precision / (precision + recall)
    f1_score = 1 - 2 * recall * precision / (precision + recall)

    return f1_score

def compute_soft_ui(labels, softmax, epsilon):
    intersection = tf.reduce_sum(labels * softmax, reduction_indices=0)
    union = tf.reduce_sum(labels + softmax, reduction_indices=0) - intersection + epsilon

    mean_iou = 1 - tf.reduce_mean(intersection / union, name='mean_iou')

    return mean_iou

# --------------------------Loss-------------------------------

