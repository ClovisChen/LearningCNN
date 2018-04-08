#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2

from utils.extract_image import *


def extract_indoor():
    video_path = '/media/bobin/Seagate/data/slam/iacas/indoor/video_1/'
    video_name = 'IMG_0030.MOV'
    num = extract_image_from_video(video_path, video_name, True)
    gene_file_list(video_path + 'file_apple_indoor.txt', num)


def extract_outdoor():
    # video_path = '/media/bobin/DATA1/SLAM/data/outdoor/apple/'
    # gene_file_list(video_path + 'file_apple_park.txt')
    video_path = '/media/bobin/Seagate/data/slam/iacas/outdoor/apple/'
    video_name = 'IMG_0027.MOV'
    extract_image_from_video(video_path, video_name)
    gene_file_list(video_path + 'file_apple_park.txt')


if __name__ == '__main__':
    extract_indoor()
