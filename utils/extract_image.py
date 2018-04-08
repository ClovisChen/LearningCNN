#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import cv2


def gene_file_list(filename, num):
    with open(filename, 'w') as f:
        for i in range(num):
            if i % 5:
                continue
            f.write('img/%.6d.png ' % i)
            f.write('img/%.6d.png\n' % i)

        f.close()


def extract_image_from_video(video_path, video_name, show_flag=False):
    cap = cv2.VideoCapture(video_path + video_name)
    ret, frame = cap.read()
    cnt = 0
    while ret:
        if show_flag:
            cv2.imshow('extract images form video', frame)
            cv2.waitKey(10)
        ret, frame = cap.read()
        if not ret:
            break
        h, w, c = frame.shape
        frame = cv2.resize(frame, (w / 2, h / 2))
        if cnt % 5:
            cnt += 1
            continue

        cv2.imwrite(video_path + "sfm/%.6d.jpg" % cnt, frame)
        # ret, frame = cap.read()
        print cnt
        cnt += 1
    return cnt