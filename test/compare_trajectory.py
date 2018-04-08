#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import EurocReader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pose.PyLie.transform as tf



def sync_trajectory(gt_traj_file, output_traj_file):
    gt_traj = EurocReader.read_ground_truth(gt_traj_file)
    out_traj = EurocReader.read_ground_truth(output_traj_file)
    gt_lens = len(gt_traj)
    out_lens = len(out_traj)
    start = 0
    pose_out_init = None
    pose0 = gt_traj[0]
    pos0 = pose0[1:4]
    quat0 = pose0[4:8]
    pose_gt_init = tf.quaternion_matrix(quat0)
    pose_gt_init[:3, 3] = pos0
    gt_stamp = float(gt_traj[0, 0] / 1e9)

    for i in range(gt_lens):
        if out_traj[i, 0] >= gt_stamp:
            pose0 = out_traj[i]
            pos0 = pose0[1:4]
            quat0 = pose0[4:8]
            pose_out_init = tf.quaternion_matrix(quat0)
            pose_out_init[:3, 3] = pos0
            start = i
            break
    gt_index = np.empty(len(out_traj), dtype=int)
    prev_index = 0
    gt_index[:start] = int(-1)
    end = out_lens
    for i in range(start, out_lens):
        if prev_index < gt_lens:
            while out_traj[i, 0] >= gt_traj[prev_index, 0] / 1e9:
                prev_index += 1
                if prev_index >= gt_lens:
                    end = i
                    break
            gt_index[i] = int(prev_index - 1)
        else:
            break

    ones_out = np.ones((out_lens, 1))
    out_abs_pos = np.concatenate((out_traj[:, 1:4], ones_out), axis=1)
    out_rel_pos = np.linalg.inv(pose_out_init).dot(out_abs_pos[start:end, :].T)

    ones_gt = np.ones((gt_lens, 1))
    gt_abs_pos = np.concatenate((gt_traj[:, 1:4], ones_gt), axis=1)
    gt_rel_pos = np.linalg.inv(pose_gt_init).dot(gt_abs_pos.T)
    err = np.zeros(end - start, dtype=int)
    err = out_rel_pos[:3, :] - gt_rel_pos[:3, gt_index[start:end]]
    err = np.sqrt(np.sum(err ** 2, axis=0) / 3)
    return gt_rel_pos, out_rel_pos, gt_index, err


def visualize_err(err):
    plt.figure()
    plt.plot(err)


def visualize_trajectory(gt_rel_pos, out_rel_pos):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(out_rel_pos[0, :], out_rel_pos[1, :], out_rel_pos[2, :], label='gt trajectory', color='red')
    ax.plot(gt_rel_pos[0, :], gt_rel_pos[1, :], gt_rel_pos[2, :], label='output trajectory', color='blue')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    # gt_file = '/home/bobin/data/euroc/V1_01_easy/state_groundtruth_estimate0/data.csv'
    # output_file = '/home/bobin/Documents/code/VIO/ygz/dw_ws/ygz-stereo-inertial/examples/output/V101.txt'
    gt_file = '/home/bobin/data/euroc/MH_01_easy/state_groundtruth_estimate0/data.csv'
    output_file_vins = '/home/bobin/Documents/ROS_WS/vin/src/VINS-Mono/config/output/MH01.txt'
    output_file_vio = '/home/bobin/Documents/code/VIO/ygz/dw_ws/ygz-stereo-inertial/examples/output/MH01.txt'
    # gt_file = '/home/bobin/data/euroc/MH_01_easy/state_groundtruth_estimate0/data.csv'
    # output_file = '/home/bobin/Documents/code/VIO/ygz/dw_ws/ygz-stereo-inertial/examples/output/MH01.txt'
    # gt_file = '/home/bobin/data/euroc/MH_02_easy/state_groundtruth_estimate0/data.csv'
    # output_file = '/home/bobin/Documents/code/VIO/ygz/dw_ws/ygz-stereo-inertial/examples/output/MH02.txt'
    # gt_file = '/home/bobin/data/euroc/MH_05_difficult/state_groundtruth_estimate0/data.csv'
    # output_file = '/home/bobin/Documents/code/VIO/ygz/dw_ws/ygz-stereo-inertial/examples/output/MH05.txt'

    gt_rel_pos_vins, out_rel_pos_vins, gt_index_vins, err_vins = sync_trajectory(gt_file, output_file_vins)
    _, out_rel_pos_vio, gt_index_vio, err_vio = sync_trajectory(gt_file, output_file_vio)
    plt.figure()
    plt.plot(err_vins, label='vins')
    plt.plot(err_vio, label='proposed method')
    plt.legend()

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.plot(out_rel_pos_vins[0, :], out_rel_pos_vins[1, :], out_rel_pos_vins[2, :], label='vins trajectory', color='red')
    ax.plot(out_rel_pos_vio[0, :], out_rel_pos_vio[1, :], out_rel_pos_vio[2, :], label='proposed method trajectory', color='green')
    ax.plot(gt_rel_pos_vins[0, :], gt_rel_pos_vins[1, :], gt_rel_pos_vins[2, :], label='ground trajectory', color='blue')
    ax.legend()
    plt.show()