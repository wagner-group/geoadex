import logging
import os
import pdb
import pickle
import time

import numpy as np
import scipy
import torch
import torch.functional as F
from lib.dataset_utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():

    idx = 1
    TOL = 1e-5
    name = 'test_nb_pca25'

    # Get logger
    log_file = name + '.log'
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s %(name)s] %(message)s')
    # Create file handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.info(log_file)

    (x_train, y_train), (_, _), (_, _) = load_mnist_all(
        data_dir='/data', val_size=0.1, shuffle=True, seed=1)
    points = x_train.view(-1, 784).numpy()

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=25)
    points = pca.fit_transform(points)

    mid = []
    nonmid = []
    nearest_point = np.copy(points[idx])
    points = np.delete(points, idx, axis=0)
    distance = np.zeros(points.shape[0]) - 1

    log.info('checking midpoints...')
    start = time.time()
    for i in range(points.shape[0]):
        midpoint = (nearest_point + points[i]) / 2
        dist = ((midpoint - points) ** 2).sum(1)
        min_dist = dist.min()
        min_idx = np.where(dist - min_dist < TOL)[0]
        dist_to_nearest_point = ((midpoint - nearest_point) ** 2).sum()
        if abs(dist_to_nearest_point - min_dist) < 1e-12 and i in min_idx and len(min_idx) == 1:
            mid.append(i)
            distance[i] = np.sqrt(min_dist)
        else:
            nonmid.append(i)
        if i % 1000 == 0:
            log.debug(i)
    end = time.time()

    log.info('runtime for checking midpoint: %.4fs', end - start)
    log.info('len of mid: %d', len(mid))
    log.info('len of non-mid: %d', len(nonmid))

    def get_polytope(points, nearest_point):
        A = np.zeros((points.shape[0], points.shape[1]))
        b = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            A[i] = points[i] - nearest_point
            b[i] = (A[i] @ (points[i] + nearest_point)) / 2
        return A, b

    A, b = get_polytope(points, nearest_point)
    A = A.astype(np.float32)
    b = b.astype(np.float32)
    assert A.shape == points.shape

    AAT = A @ A.T
    norm_A = np.diag(AAT)
    d = A.shape[0]

    num_steps = 1000
    feas = []

    x_hat = nearest_point
    # x_hat = np.zeros(A.shape[1]) + 1
    b_hat = A @ x_hat - b

    def objective(lamda):
        return - 0.5 * ((A.T @ lamda) ** 2).sum() + lamda @ b_hat

    def cga_update(lamda, g, idx_hp):
        gg = np.maximum(0, lamda + g / norm_A) - lamda
        gg[idx_hp] = g[idx_hp] / norm_A[idx_hp]
        gg_abs = np.abs(gg)
        i_star = gg_abs.argmax()
        lamda[i_star] += gg[i_star]
        g = g - gg[i_star] * AAT[:, i_star]
        return lamda, g, gg_abs

    def cga(idx_hp):
        lamda = np.zeros(d)
        g = b_hat
        for step in range(num_steps):
            lamda, g, gg = cga_update(lamda, g, idx_hp)
            if gg.max() < TOL:
                obj = objective(lamda)
                # log.info(obj)
                # log.info(step)
                return obj
        return None

    def run_all_hp():
        for i, idx_hp in enumerate(nonmid):
            obj = cga(idx_hp)
            if obj is not None:
                feas.append(idx_hp)
                distance[idx_hp] = np.sqrt(obj * 2)
            if i % 1000 == 0:
                log.debug(i)

    log.info('start solving QPs on non-midpoints...')
    start = time.time()
    run_all_hp()
    end = time.time()
    log.info('QP runtime: %.4fs', end - start)

    log.info('len of feas: %d', len(feas))
    log.info('number of edges: %d', (len(feas) + len(mid)))
    log.info('number of non-edges: %d', (len(points) - len(feas) - len(mid)))

    pickle.dump(distance, open('distance_%s.p' % name, 'wb'))


if __name__ == '__main__':
    main()
