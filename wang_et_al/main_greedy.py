import math
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as st

from knn_robustness.knn import GreedyAttack, SubsolverFactory
from knn_robustness.utils import initialize_data, initialize_params


def print_ci(mean, sem, num_trials):
    for ci in [0.9, 0.95, 0.99]:
        lo, hi = st.t.interval(ci, num_trials - 1, loc=mean, scale=sem)
        interval = mean - lo
        print(f'{ci}-confidence interval: {mean:.4f} +/- {interval:.6f}')


def get_ci(num_trials, dataset=None):

    output = {
        'dist': [],
        'runtime': []
    }
    rep = 0
    for _ in range(num_trials):
        mean_out = None
        while mean_out is None:
            mean_out = main(seed=np.random.randint(2 ** 32 - 1), dataset=dataset)
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


def main(seed=None, dataset=None):

    params = initialize_params('greedy')
    # DEBUG: handle CI and automate dataset experiments
    if seed is not None:
        params['seed'] = str(seed)
    if dataset is not None:
        params['dataset'] = dataset

    X_train, y_train, X_test, y_test = initialize_data(params)

    attack = GreedyAttack(
        X_train=X_train,
        y_train=y_train,
        n_neighbors=params.getint('n_neighbors'),
        subsolver=SubsolverFactory().create(params.get('subsolver')),
        n_far=params.getint('n_far'),
        max_trials=params.getint('max_trials'),
        min_trials=params.getint('min_trials')
    )

    count = 0
    success_notes = []
    perturbation_norms = []
    start = time.time()
    for instance, label in zip(X_test, y_test):
        if attack.predict_individual(instance) != label:
            continue
        perturbation = attack(instance)
        if perturbation is None:
            success = False
            perturbation_norm = math.inf
        else:
            success = True
            perturbation_norm = np.linalg.norm(perturbation)

        success_notes.append(success)
        perturbation_norms.append(perturbation_norm)

        details = pd.DataFrame({
            'success': success_notes,
            'perturbation': perturbation_norms
        })
        details.to_csv(os.path.join(params.get('result_dir'), 'detail.csv'))

        count += 1
        print(f'{count:03d} {success} {perturbation_norm:.7f}')
        if count >= params.getint('n_evaluate'):
            break

    runtime = time.time() - start
    print(f'Runtime (s): {runtime:.4f}')

    summary = pd.DataFrame({
        'num': [count],
        'success_rate': [details['success'].sum()/count],
        'mean': [details['perturbation'][details['success']].mean()],
        'median': [details['perturbation'][details['success']].median()],
        'runtime': [runtime],
    })

    summary.to_csv(os.path.join(params.get('result_dir'), 'summary.csv'))
    print(summary)
    print(params.getint('n_far'), params.getint('max_trials'), params.getint('min_trials'))
    print(params.get('dataset'))

    if details['success'].sum() != count:
        return None
    return details['perturbation'].mean(), runtime


if __name__ == '__main__':
    for dataset in [
        'australian',
        'covtype',
        'diabetes',
        'fourclass',
        'gaussian',
        'letter',
        'yang-fmnist'
    ]:
        print(f'===================== {dataset} =====================')
        get_ci(10, dataset=dataset)
