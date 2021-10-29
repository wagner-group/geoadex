import time

import numpy as np
import torch

from lib.dataset_utils import load_mnist_all
from lib.dknn import DKNNL2, KNNModel
from lib.dknn_attack_v2 import DKNNAttackV2
from lib.geoadex import GeoAdEx
from lib.loaders import initialize_data


def main():

    dataset_params = {
        'dataset': 'letter',
        # 'dataset': 'pendigits',
        'dataset_dir': '/home/chawin/data/',
        'random': True,
        'seed': 1,
        'partial': False,
        'label_domain': (0, 1),
    }

    upperbound = np.inf
    i = 0

    params = {
        # ======== general params ======== #
        'k': 1,
        'exact': True,
        'method': 'gca',
        'dtype': np.float32,
        'parallel': False,
        'num_cores': 32,
        'tol': 1e-7,
        'device': 'cpu',
        # ======== cert params ======== #
        'time_limit': 1000,    # time limit in seconds
        'neighbor_method': 'all',  # schemes for picking neighbors
        # 'neighbor_method': 'm_nearest',
        'm': 20,
        'save_1nn_nb': True,
        'compute_dist_to_cell': False,
        'treat_facet_as_cell': False,    # treat dist to facet as dist to cell
        'use_potential': False,
        # ======== gca params ======== #
        'max_proj_iters': 2000,
        'early_stop': True,
        'check_obj_steps': 200,
        'upperbound': upperbound,
        'div_counter': 8,
        'div_ratio': 0.999,
        'div_step': 10,
    }

    x_train, y_train, x_test, y_test = initialize_data(dataset_params)
    x_train = x_train.astype(params['dtype'])
    x_test = x_test.astype(params['dtype'])

    net_knn = KNNModel()
    knn = DKNNL2(net_knn,
                 torch.from_numpy(x_train), torch.from_numpy(y_train),
                 torch.from_numpy(x_test), torch.from_numpy(y_test),
                 ['identity'], k=params['k'],
                 num_classes=len(np.unique(y_train)), device=params['device'])

    attack = DKNNAttackV2(knn)

    def attack_batch(x, y, batch_size):
        x_adv = torch.zeros_like(x)
        total_num = x.size(0)
        num_batches = int(np.ceil(total_num / batch_size))
        for i in range(num_batches):
            begin = i * batch_size
            end = (i + 1) * batch_size
            x_adv[begin:end] = attack(
                x[begin:end], y[begin:end], 2, guide_layer=['identity'],
                m=params['k'] * 2, binary_search_steps=5, init_mode=1,
                init_mode_k=1, max_iterations=1000, learning_rate=1e-2,
                max_linf=None, thres_steps=100, check_adv_steps=200,
                initial_const=1e-1, random_start=True, verbose=False)
        return x_adv

    x_adv = attack_batch(
        torch.from_numpy(x_test[i:i + 1]).to(params['device']),
        torch.from_numpy(y_test[i:i + 1]).to(params['device']), 100)
    upperbound = np.linalg.norm(x_adv[0].detach().numpy() - x_test[i])
    print('upper bound found by a quick attack: %.4f' % upperbound)
    params['upperbound'] = upperbound
    # import pdb
    # pdb.set_trace()

    # x_adv = attack_batch(
    #     torch.from_numpy(x_test[i:i + 20]).to(params['device']),
    #     torch.from_numpy(y_test[i:i + 20]).to(params['device']), 100)
    # upperbound = ((x_adv[i:i + 20].detach().numpy() - x_test[i:i + 20]) ** 2).sum()

    query = x_test[i].flatten().astype(params['dtype'])
    label = y_test[i]

    # import pdb
    # pdb.set_trace()

    gc = GeoAdEx(x_train, y_train, params['k'], knn.indices[0], None)
    params['upperbound'] = gc.compute_potential(
        query, label, x_adv[0].detach().numpy(), params)
    start = time.time()
    out = gc.get_cert(query, label, params, k=None)
    print('Total runtime: %.2fs' % (time.time() - start))


if __name__ == '__main__':
    main()
