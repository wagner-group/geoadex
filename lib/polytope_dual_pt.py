'''Implement dual algorithms.'''
import numpy as np
import torch
import torch.nn.functional as F


def objective(lamda, A, b_hat):
    return - 0.5 * ((A.T @ lamda) ** 2).sum() + lamda @ b_hat


def find_coordinate(lamda, grad, idx_plane=None):
    # largest step size
    delta = F.relu(lamda + grad) - lamda
    if idx_plane is not None:
        delta[idx_plane] = grad[idx_plane]
    delta_abs = torch.abs(delta)
    idx = delta_abs.argmax()
    return idx, delta[idx]
    # idx = torch.randint(lamda.size(0), (1, ))
    # return idx, delta[idx].squeeze()


def gca_update(lamda, grad, AAT, idx_plane=None):
    idx, delta = find_coordinate(lamda, grad, idx_plane=idx_plane)
    # update lamda and gradient
    lamda[idx] += delta
    grad -= delta * AAT[:, idx]
    return lamda, grad, delta.abs()

    # delta = torch.max(g, - lamda)
    # # delta[idx_hp] = g[idx_hp]
    # obj_increase = delta * (g - 0.5 * delta)
    # i_star = obj_increase.argmax()
    # lamda[i_star] += delta[i_star]
    # g = g - delta[i_star] * AAT[:, i_star]
    # return lamda, g, delta.abs()


def parallel_gca_update(lamda, grad, AAT, num_partitions=10):
    # size = int(np.ceil(lamda.size(0) / num_partitions))
    # # indices = torch.zeros(size, dtype=torch.long, device=lamda.device)
    # lamda_update = torch.zeros_like(lamda)
    # grad_update = torch.zeros_like(grad)
    # for i in range(num_partitions):
    #     start, end = i * size, (i + 1) * size
    #     idx, delta = find_coordinate(
    #         lamda[start:end], grad[start:end])
    #     lamda_update[start + idx] += delta
    #     # import pdb
    #     # pdb.set_trace()
    #     grad_update -= delta * AAT[:, start + idx]
    # # return lamda + lamda_update, grad + grad_update, delta.abs()
    # return lamda + lamda_update, grad + grad_update, lamda_update.abs().max()

    size = int(np.ceil(lamda.size(0) / num_partitions))
    indices = torch.zeros(
        num_partitions, dtype=torch.long, device=lamda.device)
    deltas = torch.zeros(num_partitions, device=lamda.device)
    for i in range(num_partitions):
        start, end = i * size, (i + 1) * size
        idx, deltas[i] = find_coordinate(
            lamda[start:end], grad[start:end])
        indices[i] = start + idx
    # import pdb
    # pdb.set_trace()
    lamda[indices] += deltas
    grad -= (deltas.unsqueeze(0) * AAT[:, indices]).sum(1)
    return deltas.abs().max()


def gca(x_hat, A, AAT, b, params, idx_plane=None):

    b_hat = A @ x_hat - b
    lamda = torch.zeros(
        A.size(0), device=params['device'], dtype=params['dtype'])
    grad = b_hat.clone()

    for step in range(params['max_proj_iters']):
        lamda, grad, delta = gca_update(lamda, grad, AAT, idx_plane=idx_plane)
        if params['early_stop']:
            if step % params['check_obj_steps'] == 0:
                if objective(lamda, A, b_hat) >= params['upperbound']:
                    return None
            if delta < params['tol']:
                break

    return x_hat - lamda @ A


def parallel_gca(x_hat, A, AAT, b, params):

    b_hat = A @ x_hat - b
    lamda = torch.zeros(
        A.size(0), device=params['device'], dtype=params['dtype'])
    grad = b_hat.clone()

    for step in range(params['max_proj_iters']):
        delta = parallel_gca_update(
            lamda, grad, AAT, num_partitions=params['num_partitions'])
        if params['early_stop'] and delta < params['tol']:
            break

    return x_hat - lamda @ A


def dual_ascent(x_hat, A, AAT, b, params):
    """
    Solve max -0.5 * lamda.T @ AAT @ lamda + lamda @ b_hat.
    b_hat is defined as A @ x_hat - b.
    """

    b_hat = A @ x_hat - b
    lamda = torch.zeros(
        A.size(0), device=params['device'], dtype=params['dtype'])

    for step in range(params['max_proj_iters']):
        grad = b_hat - AAT @ lamda
        lamda = F.relu(lamda + params['step_size'] * grad)
        # NOTE: we want to compute norm of gradient but doing so at every
        # iteration can be expensive. The if statement is also structured this
        # way to avoid unnecessary compuation when early_stop is set to False.
        if params['early_stop']:
            # if step % params['check_obj_steps'] == 0:
            #     if objective(lamda, A, b_hat) >= params['upperbound']:
            #         return None
            if grad.abs().max() <= params['tol']:
                break

    return x_hat - lamda @ A
