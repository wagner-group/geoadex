import logging
import pprint
import time
from copy import deepcopy

import numpy as np
import torch
import scipy.stats as st

from lib.dknn import DKNNL2, KNNModel
from lib.dknn_attack_v2 import DKNNAttackV2
from lib.loaders import initialize_data
from lib.utils.utils import get_logger


def print_ci(mean, sem, num_trials):
    for ci in [0.9, 0.95, 0.99]:
        lo, hi = st.t.interval(ci, num_trials - 1, loc=mean, scale=sem)
        interval = mean - lo
        print(f'{ci}-confidence interval: {mean:.4f} +/- {interval:.4f}')


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
        dist, runtime = mean_out
        output['dist'].append(dist)
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


def get_precise_label(points, labels, inpt, k, num_classes):
    """
    Use this method to classify when <inpt> is close to or on multiple
    bisectors. Normal knn can be ambiguous in this case.
    """
    TOL = 1e-6

    dist = np.sum((inpt - points) ** 2, 1)
    # Find distance to the kth neighbor
    k_dist = np.sort(dist)[k - 1]
    indices = np.where(dist - k_dist < TOL)[0]

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
        num_fill = min(num_to_fill, close_counts[i])
        new_counts = deepcopy(sure_counts)
        new_counts[i] += num_fill
        close_counts_tmp = deepcopy(close_counts)
        # Fill the other classes in a way that benefits class i most
        while num_fill < num_to_fill:
            assert np.all(close_counts_tmp >= 0)
            # Get classes that can still be filled except for i
            ind = np.setdiff1d(np.where(close_counts_tmp > 0)[0], i)
            # Find class with the smallest count
            ind_to_fill = ind[new_counts[ind].argmin()]
            new_counts[ind_to_fill] += 1
            close_counts_tmp[ind_to_fill] -= 1
            num_fill += 1
        assert new_counts.sum() == k
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


def main(test_params, gc_params, sw_scale=1):

    # Set up logger
    log_name = 'sw_%s_k%d_exp%d' % (test_params['dataset'], gc_params['k'],
                                    test_params['exp'])
    log = get_logger(log_name, level=test_params['log_level'])
    log.info('\n%s', pprint.pformat(test_params))

    # Load data
    x_train, y_train, x_test, y_test = initialize_data(test_params)
    x_train = x_train.astype(gc_params['dtype'])
    x_test = x_test.astype(gc_params['dtype'])
    num_test = test_params['num_test']
    num_classes = len(np.unique(y_train))
    log.info('Training data shape: %s' % str(x_train.shape))
    log.info('Test data shape: %s' % str(x_test.shape))

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

    params = {
        'binary_search_steps': 5,
        'max_iterations': 1000,
        'thres_steps': 50,
        'check_adv_steps': 50,
    }

    for key in params:
        if key in ('binary_search_steps', 'max_iterations'):
            params[key] = int(params[key] * sw_scale)
        else:
            params[key] = int(np.ceil(params[key] / sw_scale))

    def attack_batch(x, y, batch_size, mode):
        x_adv = torch.zeros_like(x)
        total_num = x.size(0)
        num_batches = int(np.ceil(total_num / batch_size))

        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            mode_params = {
                1: {
                    'init_mode': 1,
                    'init_mode_k': 1,
                    'learning_rate': 1e-2,
                },
                2: {
                    'init_mode': 2,
                    'init_mode_k': gc_params['k'],
                    'learning_rate': 1e-1,
                },
            }[mode]
            x_adv[begin:end] = attack(
                x[begin:end], y[begin:end], 2, guide_layer=['identity'],
                m=gc_params['k'] * 2, max_linf=None, random_start=True,
                initial_const=1e-1, verbose=False, **params, **mode_params)
        return x_adv

    log.info('Finding correctly classified samples...')
    y_pred = knn.classify(torch.from_numpy(x_test[:num_test * 2]))
    ind = np.where(y_pred.argmax(1) == y_test[:num_test * 2])[0]
    ind = ind[:num_test]
    assert len(ind) == num_test

    # DEBUG: testing min distance to diff class
    # dist_all = []
    # for x, y in zip(x_test[ind], y_test[ind]):
    #     dists = np.sqrt(((x - x_train) ** 2).sum(1))
    #     dist_all.append(dists[y != y_train].min())
    # print(np.mean(dist_all))
    # assert False

    start = time.time()
    log.info('Running the heuristic attack...')
    x_adv = attack_batch(
        torch.from_numpy(x_test[ind]).to(gc_params['device']),
        torch.from_numpy(y_test[ind]).to(gc_params['device']), 100, 1)

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
            torch.from_numpy(y_test[ind]).to(gc_params['device']), 100, 2)
        log.info('Verifying the heuristic attack (2)...')
        ind_correct = classify(
            x_train, y_train, x_adv2.detach().cpu().numpy(), y_test[ind],
            gc_params, num_classes)
        upperbound2 = np.linalg.norm(x_adv2.detach().numpy() - x_test[ind], 2, 1)
        upperbound2[ind_correct] = np.inf
        ind2 = upperbound2 < upperbound
        upperbound[ind2] = upperbound2[ind2]
        x_adv[ind2] = x_adv2[ind2]
    log.info('Upper bound found by a quick attack: %s', str(upperbound))
    if np.any(upperbound > 1e9):
        log.info('Not all heuristic attacks succeed! Fix this manually.')
        return None

    runtime = time.time() - start
    log.info('Total runtime: %.2fs', runtime)
    log.info('SW mean dist.: %.4f' % np.mean(upperbound))

    # Closing log files
    handlers = log.handlers[:]
    for handler in handlers:
        handler.close()
        log.removeHandler(handler)

    return np.mean(upperbound), runtime


if __name__ == '__main__':

    gc_params = {
        'k': 7,
        'device': 'cpu',
        'dtype': np.float32,
    }

    test_params = {
        'exp': 1,
        # 'dataset': 'letter',
        # 'dataset': 'pendigits',
        # 'dataset': 'mnist',
        # 'dataset': 'gaussian',
        'dataset': 'australian',
        # 'dataset': 'cancer',
        # 'dataset': 'diabetes',
        # 'dataset': 'fourclass',
        # 'dataset': 'covtype',
        # 'dataset': 'halfmoon',
        # 'dataset': 'yang-mnist',
        # 'dataset': 'yang-fmnist',
        # 'dataset': 'ijcnn',
        'dataset_dir': '/home/chawin/space-partition-adv/data/',
        'random': True,
        'seed': 1,
        'partial': False,
        'label_domain': (1, 7),     # Only used when partial = True
        'num_test': 100,
        'log_level': logging.INFO,
        'gaussian': {
            'dim': 20,
            'dist': 0.5,
            'sd': 1.,
            'num_points': 12500,
            'test_ratio': 0.2
        }
    }

    # for i, scale in enumerate([8]):
    #     # for i, scale in enumerate([1, 2, 3, 4, 5, 6]):
    #     test_params['exp'] = i + 5
    #     main(test_params, gc_params, sw_scale=scale)

    for dataset in [
        # 'australian',
        'covtype',
        'diabetes',
        'fourclass',
        'gaussian',
        'letter',
        'yang-fmnist'
    ]:
        test_params['dataset'] = dataset
        print(f'===================== {dataset} =====================')
        get_ci(test_params, gc_params, 4, 10)
