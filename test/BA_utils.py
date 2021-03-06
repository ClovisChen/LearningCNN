from __future__ import print_function
import time
from scipy.optimize import least_squares
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import cv2


# FILE_NAME = "prob.txt"
def read_bal_data(file_name):
    with open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


def vec_camera(vec):
    f = vec[0]
    # k1 = vec[1]
    # k2 = vec[2]
    camera = np.diag([f, f, 1])
    return camera


def pose_euler_trans(vec):
    angle = vec[:3]
    trans = vec[3:]
    rot, _ = cv2.Rodrigues(angle)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = trans
    return pose


def rotate(points, rot_vecs):
    """
    Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj ** 2, axis=1)
    r = 1 + k1 * n + k2 * n ** 2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
    points_3d = params[n_cameras * 9:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def fun_ext(params, intrinsics, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    camera_params = np.concatenate((intrinsics, camera_params), axis=1)
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A

def bundle_adjustment_sparsity_ext(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A

def prettylist(l):
    return '[%s]' % ', '.join("%4.1e" % f for f in l)


def test_only_ext():
    FILE_NAME = "../data/problem-49-7776-pre.txt"
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

    # Print information
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 6 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    # x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    # f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    x0 = np.hstack((camera_params[:, 3:].ravel(), points_3d.ravel()))
    f0 = fun_ext(x0, camera_params[:, :3], n_cameras, n_points, camera_indices, point_indices, points_2d)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(f0)

    # A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    A = bundle_adjustment_sparsity_ext(n_cameras, n_points, camera_indices, point_indices)
    print("A shape: {}".format(A.shape))


    # print(A.data)

    t0 = time.time()
    res = least_squares(fun_ext, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',loss='huber',
                        args=(camera_params[:, :3], n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    print('Before:')
    print('cam0: {}'.format(prettylist(x0[0:6])))
    print('cam1: {}'.format(prettylist(x0[6:12])))

    print('After:')
    print('cam0: {}'.format(prettylist(res.x[0:6])))
    print('cam1: {}'.format(prettylist(res.x[6:12])))

    np.savetxt('param.txt', res.x)

    plt.subplot(212)
    plt.plot(res.fun)
    plt.show()


def test_all():
    FILE_NAME = "../data/problem-49-7776-pre.txt"
    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)

    # Print information
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    # x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    # f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(f0)

    # A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    print("A shape: {}".format(A.shape))

    # print(A.data)

    t0 = time.time()
    res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',loss='huber',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    print('Before:')
    print('cam0: {}'.format(prettylist(x0[0:9])))
    print('cam1: {}'.format(prettylist(x0[9:18])))

    print('After:')
    print('cam0: {}'.format(prettylist(res.x[0:9])))
    print('cam1: {}'.format(prettylist(res.x[9:18])))

    np.savetxt('param.txt', res.x)

    plt.subplot(212)
    plt.plot(res.fun)
    plt.show()

if __name__ == '__main__':
    # test_all()
    test_only_ext()