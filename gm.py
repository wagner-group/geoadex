'''
Some of the code is taken from 
https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
'''

import logging
import pickle
import pprint
import time

import faiss
import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from lib.dknn import DKNNL2, KNNModel
from lib.dknn_attack_v2 import DKNNAttackV2
from lib.geocert import GeoCert
from lib.loaders import initialize_data
from lib.utils.utils import get_logger


def kld(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term = np.trace(iS1 @ S0)
    det_term = np.log(np.linalg.det(S1) / np.linalg.det(S0))
    quad_term = diff.T @ iS1 @ diff
    return .5 * (tr_term + det_term + quad_term - N)


def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample` """
    norms = np.linalg.norm(sample-point, axis=1)
    return np.sort(norms)[k]


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert(len(s1.shape) == len(s2.shape) == 2)
    # Check dimensionality of sample is identical
    assert(s1.shape[1] == s2.shape[1])


def naive_estimator(s1, s2, k=1):
    """ KL-Divergence estimator using brute-force (numpy) k-NN
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    D = np.log(m / (n - 1))
    d = float(s1.shape[1])

    for p1 in s1:
        nu = knn_distance(p1, s2, k-1)  # -1 because 'p1' is not in 's2'
        rho = knn_distance(p1, s1, k)
        if rho < 1e-9:
            n -= 1
            continue
        D += (d / n) * np.log(nu / rho)
    return D


def main(test_params, gc_params):

    # Load data
    x_train, y_train, x_test, y_test = initialize_data(test_params)
    x_train = x_train.astype(gc_params['dtype'])
    x_test = x_test.astype(gc_params['dtype'])
    num_test = test_params['num_test']

    labels = np.unique(y_train)
    num_labels = len(labels)
    dim = x_train.shape[-1]
    means = np.zeros((num_labels, dim))
    covariances = np.zeros((num_labels, dim, dim))

    # for i, label in enumerate(labels):
    #     gm = GaussianMixture(n_components=1, covariance_type='full')
    #     gm.fit(x_train[y_train == label])
    #     means[i] = gm.means_[0]
    #     covariances[i] = gm.covariances_[0]

    # dists = np.zeros((num_labels, num_labels))
    # for i in range(num_labels):
    #     for j in range(num_labels):
    #         if i == j:
    #             # Intentionally set KLD of the same class to infinity
    #             dists[i, j] = np.inf
    #             continue
    #         dists[i, j] = kld(means[i], covariances[i],
    #                           means[j], covariances[j])

    # import pdb
    # pdb.set_trace()

    dists = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        for j in range(num_labels):
            if i == j:
                # Intentionally set KLD of the same class to infinity
                dists[i, j] = np.inf
                continue
            if np.sum(y_train == i) < 100 or np.sum(y_train == j) < 100:
                dists[i, j] = np.inf
                continue
            dists[i, j] = naive_estimator(
                x_train[y_train == i], x_train[y_train == j], k=5)

    print(test_params['dataset'])
    print('Mean of minimum KLD: %.4f' % dists.min(1).mean())


if __name__ == '__main__':

    test_params = {
        'exp': 1,
        # 'dataset': 'letter',
        # 'dataset': 'pendigits',
        # 'dataset': 'mnist',
        # 'dataset': 'gaussian',
        # 'dataset': 'australian',
        # 'dataset': 'cancer',
        'dataset': 'diabetes',
        # 'dataset': 'fourclass',
        # 'dataset': 'yang-mnist',
        # 'dataset': 'covtype',
        # 'dataset': 'halfmoon',
        # 'dataset': 'yang-fmnist',
        # 'dataset': 'ijcnn',
        'dataset_dir': '/home/chawin/data/',
        'random': True,
        'seed': 1,
        'partial': False,
        'label_domain': (1, 7),     # Only used when partial = True
        'num_test': 100,
        'init_ub': True,
        # 'init_ub': False,
        # 'log_level': logging.DEBUG,
        'log_level': logging.INFO,
        'gaussian': {
            'seed': 1,
            'dim': 20,
            'dist': 0.3,
            'sd': 1.,
            'num_points': 12500,
            'test_ratio': 0.2
        }
    }

    gc_params = {
        # ======== general params ======== #
        'k': 3,
        'exact': True,
        'method': 'gca',
        'dtype': np.float32,
        'parallel': False,
        'num_cores': 32,
        'tol': 1e-7,
        'device': 'cpu',
        # ======== cert params ======== #
        'time_limit': 100,    # time limit in seconds
        # 'neighbor_method': 'all',  # schemes for picking neighbors
        'neighbor_method': 'm_nearest',
        'm': 40,
        'save_1nn_nb': False,
        'compute_dist_to_cell': True,
        # 'compute_dist_to_cell': False,
        'treat_facet_as_cell': False,    # treat dist to facet as dist to cell
        'use_potential': False,
        'index': 'flat',
        # 'index': 'lsh',
        # ======== gca params ======== #
        'max_proj_iters': 2000,
        'max_proj_iters_verify': 10000,
        'early_stop': True,
        'check_obj_steps': 200,
        'upperbound': np.inf,
        'div_counter': 8,
        'div_ratio': 0.999,
        'div_step': 10,
    }

    main(test_params, gc_params)
