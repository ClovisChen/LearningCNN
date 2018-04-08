import cv2
import numpy as np


def test_HoghLine():
    img = cv2.imread('/home/bobin/Desktop/68140438.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 50, 100, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

    for line in lines[0]:
        x1, y1, x2, y2 = line
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow('edge', edges)
    cv2.imshow('lines', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    test_HoghLine()
