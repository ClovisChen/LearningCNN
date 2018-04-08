#!/usr/bin/env python
# -*- coding: utf-8 -*-are not covered by the UCLB ACP-A Licence,

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def bilinear_sampler_1d_h(input_images, x_offset, wrap_mode='border', name='bilinear_sampler', **kwargs):
    '''
    一维双线性采样: x_offset--输入X上偏移量的图
    重复函数 : 先将一维的x后面扩展一个维度, 然后在扩展的维度上复制相应的值, 随后将其转成一维的值, exsamples:[1,2,3] --> [1,1,2,2,3,3]
    '''
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y): #插值函数
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            # 如果包围方式是border, 那么边界长度是1, 在h和w维两侧加一排0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            # 修剪偏移量x, 让它在0到width-1+2*edge_size之间(因为偏移量不能太大,要小于等于padding之后).
            x = tf.clip_by_value(x, 0.0,  _width_f - 1 + 2 * _edge_size)

            # 向下取整x,y然后x加1向上取整x
            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            # 将向下取整的x y变成整数, 向上取整的x不能大于padding之后的宽度减1
            # cast: 类型转换
            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)
            # 第二维也就是宽度维的宽是padding之后的宽
            dim2 = (_width + 2 * _edge_size)
            # 第一维也就是图像维的宽是padding之后的分辨率
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            # 计算偏移量索引的基,先得到[0,1,2,...,batch],再将它乘宽度,变成
            # [0,dim1,2*dim1,...,batch*dim1],然后重复原图分辨率,变成
            # [0,0,......,0,dim1,dim1,......,dim1,2*dim1,2*dim1,......,2*dim1 . . batch * dim, batch * dim, ......, batch * dim]
            # 这样就变成基底了,表达的是有batch个图的基
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            # 将y的偏移乘以dim2,也就是乘以宽度,这样就得到加上y之后的基
            # y0是[0,0,...,0,1,1,....,1, . . h + 2 * e, h + 2 * e, ..., h + 2 * e]
            # 乘了dim2之后变成
            # [0, 0, ..., 0, w+2*e, w+2*e, ..., w+2*e, . . (h + 2 * e) * (w + 2 * e), ..., (h + 2 * e) * (w + 2 * e)]
            # 加上base之后得到了考虑了batch,height之后的索引
            base_y0 = base + y0 * dim2
            # 这个索引加上向上下取整的x索引和向上取整的x索引就得到了现在点的左侧点和右侧点
            idx_l = base_y0 + x0
            idx_r = base_y0 + x1
            # 将图变成[batch*w*h,channel]的形状
            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))
            # 利用tf.gather根据左右侧点的索引重新排列图,得到重排之后的左右像素
            pix_l = tf.gather(im_flat, idx_l)
            pix_r = tf.gather(im_flat, idx_r)
            # 计算双线性差值的系数x1-1和x-x0
            weight_l = tf.expand_dims(x1_f - x, 1)
            weight_r = tf.expand_dims(x - x0_f, 1)
            # 利用双线性差值方法计算像素值
            return weight_l * pix_l + weight_r * pix_r

    # get_disp函数生成视差图后,调用插值函数获得更好的图.
    def _transform(input_images, x_offset):
        '''
        转换函数首先调用meshgrid生成关于X轴和Y轴的索引
        exsamples:
        假设_width=3，经过linspace(0.0,_width_f-1.0,_width)是[ 0., 1., 2.]。height同理
        >>> x = tf.linspace(0.0, 2.0, 3)
        >>> sess.run(x)
        array([0., 1., 2. ], dtype = float32)
        >>> x = tf.linspace(0.0, 2.0, 3)
        >>> y = tf.linspace(0.0, 4.0, 5)
        >>> x_t, y_t = tf.meshgrid(x, y)
        >>> sess.run(x_t)
        array([0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.],
            [0., 1., 2.]], dtype=float32)
        >>> sess.run(y_t)
        array([0., 0., 0.],
            [1., 1., 1.],
            [2., 2., 2.],
            [3., 3., 3.],
            [4., 4., 4.]], dtype=float32)
        >>> x_t_flat = tf.reshape(x_t, (1, -1))
        >>> y_t_flat = tf.reshape(y_t, (1, -1))
        >>> sess.run(x_t_flat)
        array([[0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.]], dtype=float32)
        >>> sess.run(y_t_flat)
        array([[0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3., 4., 4., 4.]], dtype=float32)
        >>> x_t_flat = tf.tile(x_t_flat, tf.stack([2,1]))
        >>> sess.run(x_t_flat)
        arraay([[0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.], [0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.]], dtype=float32)
        >>> x_t_flat = tf.reshape(x_t_flat, (1, -1))
        >>> sess.run(x_t_flat)
        array([[0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2., 0., 1., 2.]], dtype=float32)
        '''
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])

            x_t_flat = x_t_flat + tf.reshape(x_offset, [-1]) * _width_f

            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))

            return output


    with tf.variable_scope(name):
        '''
        [num_batch, height, width, num_channels]
        '''
        _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[1]
        _width        = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output
