'''Implement dual algorithms.'''
import numpy as np
import torch
import torch.nn.functional as F


def objective(lamda, A, b_hat):
    return - 0.5 * ((A.T @ lamda) ** 2).sum() + lamda @ b_hat


def find_coordinate(lamda, grad, idx_plane=None):
    # largest step size
    delta = np.maximum(0, lamda + grad) - lamda
    if idx_plane is not None:
        delta[idx_plane] = grad[idx_plane]
    delta_abs = np.abs(delta)
    idx = delta_abs.argmax()
    return idx, delta[idx]


def gca_update(lamda, grad, AAT, idx_plane=None):
    idx, delta = find_coordinate(lamda, grad, idx_plane=idx_plane)
    # update lamda and gradient
    lamda[idx] += delta
    grad -= delta * AAT[:, idx]
    # return lamda, grad, abs(delta)
    return lamda, grad, delta, idx


def gca(x_hat, A, AAT, b, params, idx_plane=None, ub=None):

    if ub is None:
        ub = params['upperbound']
    ub = 0.5 * ub ** 2

    b_hat = A @ x_hat - b
    lamda = np.zeros(A.shape[0], dtype=params['dtype'])
    grad = np.copy(b_hat)
    div_counter = 0
    min_obj = np.inf

    # import time
    # start = time.time()
    for step in range(params['max_proj_iters']):
        lamda, grad, delta, idx = gca_update(
            lamda, grad, AAT, idx_plane=idx_plane)
        # lamda, grad, delta = gca_update(lamda, grad, AAT, idx_plane=idx_plane)
        # if params['stop'] and step % 100 == 0:
        #     obj = - delta * AAT[idx, :] @ lamda + \
        #         0.5 * delta ** 2 + delta * b_hat[idx]
        #     # import pdb
        #     # pdb.set_trace()
        #     print(delta, obj)
        #     # print(delta)

        # TODO: need a better way to check divergence
        # (1) count number of times delta increases w/ reset
        # if delta > params['div_ratio'] * min_delta and not params['stop']:
        #     div_counter += 1
        #     if div_counter >= params['div_counter']:
        #         return None
        # else:
        #     min_delta = delta
        #     # div_counter = 0

        # (2) count number of times delta increases w/o reset
        # if delta > params['div_ratio'] * min_delta:
        #     div_counter += 1
        #     if div_counter >= params['div_counter']:
        #         return None
        # min_delta = delta

        # early stop if the objective diverges or does not increase fast enough
        # because it means that dual is unbounded and so primal is infeasible.
        if step < params['div_step']:
            # compute increase in objective
            obj_inc = - delta * AAT[idx, :] @ lamda + \
                0.5 * delta ** 2 + delta * b_hat[idx]
            if obj_inc > params['div_ratio'] * min_obj:
                div_counter += 1
                if div_counter > params['div_counter']:
                    # if params['stop']:
                    #     import pdb
                    #     pdb.set_trace()
                    # print('diverge ', step)
                    return None
            else:
                div_counter = 0
                min_obj = obj_inc

        if params['early_stop']:
            if step % params['check_obj_steps'] == 0:
                # if dual objective is already larger than upper bound,
                # we can terminate early
                if objective(lamda, A, b_hat) >= ub:
                    # if params['stop']:
                    #     import pdb
                    #     pdb.set_trace()
                    # print('upperbound ', step)
                    return None
                # TODO: if optimality gap is
                if abs(delta) < params['tol']:
                    # break
                    # if break from delta check, can skip kkt check
                    return x_hat - lamda @ A

    # compute primal residual and complementary slackness
    res = A @ (x_hat - lamda @ A) - b
    cs = np.abs(res * lamda)
    if res.max() > params['tol'] and cs.max() > params['tol']:
        # if params['stop']:
        #     import pdb
        #     pdb.set_trace()
        # print('kkt ', step)
        return None
    return x_hat - lamda @ A
