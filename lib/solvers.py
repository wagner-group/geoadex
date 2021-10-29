import time

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F

TOL = 1e-6


def solve_feas(hplanes, idx, method="farkas"):
    """
    Check that the hyperplane at index <idx> of <hplanes> is "neccessary," or
    i.e. it intersects and is a facet of the polytope defined by <hplanes>.
    """
    if method == "farkas":
        cat_hplanes = np.concatenate(
            [hplanes, - hplanes[idx][np.newaxis, :]], axis=0)
        A = - cat_hplanes[:, :-1]
        b = cat_hplanes[:, -1]
        out = solve_farkas(A, b)
    else:
        raise NotImplementedError('method no implemented.')
    return out


def solve_farkas(A, b):
    """Solve feasibility problem using Farkas' Lemma."""

    y = cp.Variable(A.shape[0])
    constraints = [A.T @ y == 0, y >= 0]
    obj = cp.Minimize(b @ y)
    prob = cp.Problem(obj, constraints)
    start = time.time()
    prob.solve(solver=cp.MOSEK)
    # prob.solve(solver=cp.ECOS)
    # prob.solve()
    end = time.time()
    print(end - start)
    # import pdb
    # pdb.set_trace()

    return prob.value >= - TOL


def solve_qp(x_hat, A, b, params, x_init=None):

    # TODO: support torch?
    x = cp.Variable(x_hat.size())
    constraints = [A @ x <= b]
    if params['clip'] is not None:
        constraints += [x >= params['clip'][0], x <= params['clip'][1]]
    # obj = cp.Maximize(np.ones(784) @ x)
    obj = cp.Minimize(cp.sum((x - x_hat) ** 2))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.MOSEK)
    # prob.solve()

    return x.value
