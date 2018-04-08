import numpy as np

def load_trajectory(fname):
    file = open(fname)
    data = file.read()
    lines = data.split("\n")
    lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
             len(line) > 0 and line[0] != "#"]
    trajectory = dict()
    for item in lists:
        trajectory[float(item[0])] = np.float64(item[3:])
    return trajectory


def load_image_path(fname):
    file = open(fname)
    data = file.read()
    lines = data.split("\n")
    lists = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if
             len(line) > 0 and line[0] != "#"]
    images = dict()
    for item in lists:
        images[float(item[0])] = [item[1], item[3]]
    return images
