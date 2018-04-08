#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading


def worker(num):
    while True:
        print num


if __name__ == '__main__':
    threads = []
    for i in range(1):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
