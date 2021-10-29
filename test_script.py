import logging
import pickle
import pprint
import time
from copy import deepcopy

import faiss
import numpy as np
import torch
import scipy.stats as st

from lib.dknn import DKNNL2, KNNModel
from lib.dknn_attack_v2 import DKNNAttackV2
from lib.geoadex import GeoAdEx
from lib.loaders import initialize_data
from lib.utils.utils import get_logger


def print_ci(mean, sem, num_trials):
    for ci in [0.9, 0.95, 0.99]:
        lo, hi = st.t.interval(ci, num_trials - 1, loc=mean, scale=sem)
        interval = mean - lo
        print(f'{ci}-confidence interval: {mean:.4f} +/- {interval:.6f}')


def get_ci(test_params, gc_params, scale, num_trials):

    output = {
        'dist': [],
        'runtime': []
    }
    rep = 0
    for _ in range(num_trials):
        mean_out = None
        while mean_out is None:
            test_params['seed'] = np.random.randint(2 ** 32 - 1)
            mean_out = main(test_params, gc_params, sw_scale=scale)
            rep += 1
            assert rep < num_trials * 2
        dist, exit_code, runtime = mean_out
        output['dist'].append(np.mean(dist))
        output['runtime'].append(runtime)

    print(output)
    print('Distance')
    mean = np.mean(output['dist'])
    sem = st.sem(output['dist'])
    print_ci(mean, sem, num_trials)

    print('Runtime')
    mean = np.mean(output['runtime'])
    sem = st.sem(output['runtime'])
    print_ci(mean, sem, num_trials)


def get_precise_label(points, labels, input, k, num_classes):
    """
    Use this method to get the final prediction when `input` is close to or on 
    multiple bisectors. Normal k-NN classifiers can be ambiguous in this case.
    Specifically, we split neighbors into two groups: (1) "sure" = unambiguous 
    neighbors, well closer to input than the k-th neighbor, (2) "close" = 
    ambiguous neighbors that are about the same distance away as the k-th 
    neighbor. These "close" neighbors usually border one another including the
    k-th neighbor. The final prediction includes all labels that are possible
    given any combination of the neighbors.
    """
    TOL = 1e-6

    dist = np.sum((input - points) ** 2, 1)
    # Find distance to the kth neighbor
    k_dist = np.sort(dist)[k - 1]
    indices = np.where(dist - k_dist < TOL)[0]

    # Splitting neighbors into sure and close groups
    close_indices = np.where(np.abs(dist - k_dist) < TOL)[0]
    sure_indices = np.setdiff1d(indices, close_indices)
    close_labels = labels[close_indices]
    sure_labels = labels[sure_indices]
    close_counts = np.bincount(close_labels, minlength=num_classes)
    sure_counts = np.bincount(sure_labels, minlength=num_classes)

    num_to_fill = k - sure_counts.sum()
    # If number of sure counts is k, then we are done
    assert num_to_fill >= 0
    if num_to_fill == 0:
        max_count = sure_counts.max()
        return np.where(sure_counts == max_count)[0]

    y_pred = []
    for i in range(num_classes):
        # Fill class i as much as possible first
        num_fill = min(num_to_fill, close_counts[i])
        new_counts = deepcopy(sure_counts)
        new_counts[i] += num_fill
        close_counts_tmp = deepcopy(close_counts)
        # Fill the other classes in a way that benefits class i most
        while num_fill < num_to_fill:
            assert np.all(close_counts_tmp >= 0)
            # Get classes that can still be filled except for i
            ind = np.setdiff1d(np.where(close_counts_tmp > 0)[0], i)
            # Find class with the smallest count and add to it
            ind_to_fill = ind[new_counts[ind].argmin()]
            new_counts[ind_to_fill] += 1
            close_counts_tmp[ind_to_fill] -= 1
            num_fill += 1
        assert new_counts.sum() == k
        # Check if class i can be the prediction
        max_count = new_counts.max()
        if new_counts[i] == max_count:
            y_pred.append(i)
    return np.array(y_pred)


def classify(x_train, y_train, x_test, y_test, gc_params, num_classes):
    ind = []
    assert len(x_test) == len(y_test)
    for i in range(len(x_test)):
        label = get_precise_label(
            x_train, y_train, x_test[i], gc_params['k'], num_classes)
        if y_test[i] in label and len(label) == 1:
            ind.append(i)
    return ind


def main(test_params, gc_params, sw_scale=1.):

    # Set up logger
    log_name = '%s_k%d_exp%d' % (test_params['dataset'], gc_params['k'],
                                 test_params['exp'])
    log = get_logger(log_name, level=test_params['log_level'])
    log.info('\n%s', pprint.pformat(test_params))
    log.info('\n%s', pprint.pformat(gc_params))

    # Load data
    x_train, y_train, x_test, y_test = initialize_data(test_params)
    x_train = x_train.astype(gc_params['dtype'])
    x_test = x_test.astype(gc_params['dtype'])
    num_test = test_params['num_test']
    num_classes = len(np.unique(y_train))
    log.info('Training data shape: %s' % str(x_train.shape))
    log.info('Test data shape: %s' % str(x_test.shape))

    # print(np.linalg.norm(x_train, axis=1).mean())
    # assert False
    # import pdb; pdb.set_trace()

    # DEBUG
    # from scipy.spatial import Voronoi
    # start = time.time()
    # vor = Voronoi(x_train)
    # log.info('Time for building a Voronoi digram: %ds', time.time() - start)
    # return

    log.info('Setting up a quick attack for computing loose upperbound...')
    net_knn = KNNModel()
    knn = DKNNL2(net_knn,
                 torch.from_numpy(x_train), torch.from_numpy(y_train),
                 torch.from_numpy(x_test), torch.from_numpy(y_test),
                 ['identity'], k=gc_params['k'],
                 num_classes=num_classes,
                 device=gc_params['device'])

    attack = DKNNAttackV2(knn)

    def attack_batch(x, y, batch_size, mode, scale=1):
        x_adv = torch.zeros_like(x)
        total_num = x.size(0)
        num_batches = int(np.ceil(total_num / batch_size))
        sw_params = {
            'm': gc_params['k'] * 2,
            'guide_layer': ['identity'],
            'binary_search_steps': int(5 * scale),
            'max_linf': None,
            'initial_const': 1e-1,
            'random_start': True,
            'verbose': False,
        }
        for i in range(num_batches):
            begin, end = i * batch_size, (i + 1) * batch_size
            if mode == 1:
                x_adv[begin:end] = attack(x[begin:end], y[begin:end], 2,
                                          init_mode=1,
                                          init_mode_k=1,
                                          max_iterations=int(1000 * scale),
                                          learning_rate=1e-2,
                                          thres_steps=int(100 / scale),
                                          check_adv_steps=int(200 / scale),
                                          **sw_params)
            else:
                x_adv[begin:end] = attack(x[begin:end], y[begin:end], 2,
                                          init_mode=2,
                                          init_mode_k=gc_params['k'],
                                          max_iterations=int(2000 * scale),
                                          learning_rate=1e-1,
                                          thres_steps=int(50 / scale),
                                          check_adv_steps=int(50 / scale),
                                          **sw_params)
        return x_adv

    log.info('Finding correctly classified samples...')
    y_pred = knn.classify(torch.from_numpy(x_test[:num_test * 2]))
    ind = np.where(y_pred.argmax(1) == y_test[:num_test * 2])[0]
    ind = ind[:num_test]
    assert len(ind) == num_test

    start = time.time()
    if test_params['init_ub']:
        log.info('Running the heuristic attack...')
        x_adv = attack_batch(
            torch.from_numpy(x_test[ind]).to(gc_params['device']),
            torch.from_numpy(y_test[ind]).to(gc_params['device']),
            100, 1, scale=sw_scale)

        # Verify that x_adv is adversarial
        log.info('Verifying the heuristic attack...')
        ind_correct = classify(
            x_train, y_train, x_adv.detach().cpu().numpy(), y_test[ind],
            gc_params, num_classes)
        log.info('Success rate of the heuristic attack (1): '
                 f'{(1 - len(ind_correct) / num_test):.2f}')
        upperbound = np.linalg.norm(x_adv.detach().numpy() - x_test[ind], 2, 1)
        upperbound[ind_correct] = np.inf

        # Re-run the heuristic attack with <init_mode> 2 if some <x_adv> are
        # not successful
        if len(ind_correct) > 0:
            log.info('Running the heuristic attack (2)...')
            x_adv2 = attack_batch(
                torch.from_numpy(x_test[ind]).to(gc_params['device']),
                torch.from_numpy(y_test[ind]).to(gc_params['device']),
                100, 2, scale=sw_scale)
            log.info('Verifying the heuristic attack (2)...')
            ind_correct = classify(
                x_train, y_train, x_adv2.detach().cpu().numpy(), y_test[ind],
                gc_params, num_classes)
            upperbound2 = np.linalg.norm(x_adv2.detach().numpy() - x_test[ind], 2, 1)
            upperbound2[ind_correct] = np.inf
            ind2 = upperbound2 < upperbound
            upperbound[ind2] = upperbound2[ind2]
            x_adv[ind2] = x_adv2[ind2]
        log.info(f'Upper bound found by a quick attack: {upperbound}')
        if np.any(upperbound > 1e9):
            log.info('Not all heuristic attacks succeed! Fix this manually.')
            return None
    else:
        upperbound = None
        log.info('Skipping the heuristic attack.')

    log.info('Setting up GeoCert...')
    d = x_train.shape[1]
    if gc_params['index'] == 'flat':
        approx_index = faiss.IndexFlatL2(d)
    elif gc_params['index'] == 'lsh':
        # n_bits = 2 * d
        n_bits = 20
        approx_index = faiss.IndexLSH(d, n_bits)
        approx_index.train(x_train)
    else:
        raise NotImplementedError('Index not implemented.')
    approx_index.add(x_train)

    gc = GeoAdEx(x_train, y_train, gc_params['k'], knn.indices[0], log,
                 approx_index=approx_index)
    log.info('Start running GeoCert...')
    dist, adv_out, exit_code = [], [], []
    for i, idx in enumerate(ind):
        log.info(f'# ==================== SAMPLE {i} =================== #')
        query = x_test[idx].flatten().astype(gc_params['dtype'])
        label = y_test[idx]

        if test_params['init_ub']:
            gc_params['upperbound'] = upperbound[i]
        else:
            gc_params['upperbound'] = np.inf
        log.info(f'Upper bound: {gc_params["upperbound"]:.4f}')
        out = gc.get_cert(query, label, gc_params, k=None)
        adv_out.append(out[0])
        dist.append(out[1])
        exit_code.append(out[2])

    # Filter out failed samples
    dist = [d for d in dist if d < np.inf]

    runtime = time.time() - start
    log.info(f'Total runtime: {runtime:.2f}s')
    log.info(f'mean: {np.mean(dist):.4f}, median: {np.median(dist):.4f}, all: {dist}')
    log.info(f'exit code: {exit_code}')
    pickle.dump([exit_code, dist, upperbound], open(f'save/{log_name}.p', 'wb'))
    log.info('Exit code: %d, %d, %d.' % (
        np.sum(0 == np.array(exit_code)), np.sum(1 == np.array(exit_code)),
        np.sum(2 == np.array(exit_code))))
    if upperbound is not None:
        log.info(f'Init ub: {np.mean(upperbound):.4f}')

    ind_correct = classify(x_train, y_train,
                           np.stack(adv_out, axis=0), y_test[ind],
                           gc_params, num_classes)
    print(len(ind_correct))
    print(ind_correct)

    # Closing log files
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)

    return dist, exit_code, runtime


if __name__ == '__main__':

    test_params = {
        'exp': 1,
        # 'dataset': 'letter',
        # 'dataset': 'pendigits',
        # 'dataset': 'mnist',
        # 'dataset': 'gaussian',
        # 'dataset': 'australian',
        # 'dataset': 'cancer',
        # 'dataset': 'diabetes',
        # 'dataset': 'fourclass',
        # 'dataset': 'covtype',
        # 'dataset': 'halfmoon',
        # 'dataset': 'yang-mnist',
        # 'dataset': 'yang-fmnist',
        # 'dataset': 'ijcnn',
        # 'dataset_dir': '/global/home/users/chawins/space-partition-adv/data/',
        'dataset_dir': '/home/chawin/space-partition-adv/data/',
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
            'dim': 20,
            'dist': 0.5,
            'sd': 1.,
            'num_points': 12500,
            'test_ratio': 0.2
        }
    }

    gc_params = {
        # ======= general params ======== #
        'k': 3,
        'method': 'gca',
        'dtype': np.float32,
        'parallel': False,
        'num_cores': 32,        # TODO: used with parallel
        'device': 'cpu',
        # ======== cert params ======== #
        'time_limit': 100,    # time limit in seconds
        # 'neighbor_method': 'all',  # schemes for picking neighbors
        'neighbor_method': 'm_nearest',
        'm': 20,
        'save_1nn_nb': False,        # should be set to False
        'compute_dist_to_cell': True,
        # 'compute_dist_to_cell': False,
        'treat_facet_as_cell': False,    # treat dist to facet as dist to cell
        'use_potential': False,     # DEPRECATED
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
        'tol': 1e-7,
    }

    for dataset in [
        # 'australian',
        # 'covtype',
        'diabetes',
        # 'fourclass',
        # 'gaussian',
        # 'letter',
        # 'yang-fmnist'
    ]:
        print(f'===================== {dataset} =====================')
        test_params['dataset'] = dataset
        get_ci(test_params, gc_params, 2, 1)

    # for i, tl in enumerate([5, 10, 20, 40, 80]):
    #     gc_params['time_limit'] = tl
    #     test_params['exp'] = 90 + i
    #     main(test_params, gc_params, sw_scale=1)
