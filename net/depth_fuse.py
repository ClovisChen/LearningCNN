#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import settings


class DepthMap():
    def __init__(self, depth):
        self.depth_map = depth
        h, w = depth.shape
        self.depth_x_grad = cv2.Sobel(self.depth_map, cv2.CV_64F, 1, 0, ksize=3)
        self.depth_y_grad = cv2.Sobel(self.depth_map, cv2.CV_64F, 0, 1, ksize=3)
        self.ks_map = np.ones((h, w))
        self.bias_map = np.ones((h, w))

    def fuse_map_point(self, map_point):
        u, v, d = map_point
        kz = settings.depth_kernel_size
        xx1d = np.arange(-kz, kz + 1)
        xx, yy = np.meshgrid(xx1d, xx1d)
        sigma_1 = settings.depth_map_sigma_1
        sigma_2 = settings.depth_map_sigma_2
        sigma_3 = settings.depth_map_sigma_3

        depth_s = self.depth_map[v, u]
        diff = d - depth_s
        local_block = self.depth_map[v - kz: v + 1 + kz, u - kz: u + 1 + kz]

        log_k = - np.sqrt(xx * xx + yy * yy) / sigma_1
        local_weight_1 = np.exp(log_k)
        print log_k
        print local_weight_1

        uv_depth_x = self.depth_x_grad[v, u]
        uv_depth_y = self.depth_y_grad[v, u]

        local_depth_x = self.depth_x_grad[v - kz: v + 1 + kz, u - kz: u + 1 + kz]
        local_depth_y = self.depth_y_grad[v - kz: v + 1 + kz, u - kz: u + 1 + kz]

        diff_local_depth_x = np.abs(local_depth_x - uv_depth_x) + sigma_2
        diff_local_depth_y = np.abs(local_depth_y - uv_depth_y) + sigma_2
        local_weight_2 = 1.0 / diff_local_depth_x * 1.0 / diff_local_depth_y

        local_weight_3 = sigma_3 + np.exp(-np.abs(local_block - depth_s + local_depth_x * xx))
        local_weight_4 = sigma_3 + np.exp(-np.abs(local_block - depth_s + local_depth_y * yy))

        ## normalize the weights
        size = local_weight_1.shape
        local_weight_sum = np.zeros(size)
        local_weight_sum += local_weight_1
        local_weight_sum += local_weight_2
        local_weight_sum += local_weight_3
        local_weight_sum += local_weight_4

        local_weight_array = [local_weight_1, local_weight_2, local_weight_3, local_weight_4]

        # local_weight = np.zeros(size)
        local_weight = local_weight_sum - np.min(local_weight_sum)
        local_weight /= np.sum(local_weight_sum) - np.min(local_weight_sum)

        return local_weight, local_weight_array
        # local_block = local_weight * (local_block + diff)


def test_fuse_depth():
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import matplotlib.pyplot as plt

    depth_file = '/home/bobin/data/rgbd/tum/rgbd_dataset_freiburg1_xyz/depth/1305031102.160407.png'
    depth_map = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED) / 1000.0
    map_point = [500, 400, 18.0]
    depth_fuse = DepthMap(depth_map)
    local_weight, local_weight_array = depth_fuse.fuse_map_point(map_point)

    height, width = depth_map.shape
    xx1d = np.arange(width)
    yy1d = np.arange(height)
    xx, yy = np.meshgrid(xx1d, yy1d)

    plt.figure()
    plt.subplot(221)
    plt.imshow(local_weight_array[0])
    plt.subplot(222)
    plt.imshow(local_weight_array[1])
    plt.subplot(223)
    plt.imshow(local_weight_array[2])
    plt.subplot(224)
    plt.imshow(local_weight_array[3])

    plt.figure()
    plt.imshow(local_weight)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, yy, depth_map, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    test_fuse_depth()
