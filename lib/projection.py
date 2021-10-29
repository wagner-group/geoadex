import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from .polytope_dual import gca
from .solvers import solve_qp


def proj_ball(x, z, r, p):
    """Project <x> onto L<p> ball centered at <z> with radius <r>."""
    if p == 1:
        return z + proj_l1_ball(x - z, r)
    elif p == 2:
        if x.dim() == 1:
            return z + torch.renorm((x - z).unsqueeze(0), 2, 0, r).squeeze()
        return z + torch.renorm(x - z, 2, 0, r)
    elif p == np.inf:
        return z + torch.clamp(x - z, - r, r)
    else:
        raise NotImplementedError('This choice of p-norm is not implemented.')


def proj_l1_ball(v, r):
    """
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    dim = v.dim()
    assert dim <= 2

    if dim == 1:
        if v.norm(1, -1) <= r:
            return v
        u = v.abs()
        sgn = v.sign()
        return sgn * proj_prob_simplex(u, r)

    idx = (v.norm(1, -1) > r).nonzero(as_tuple=True)
    if len(idx[0]) == 0:
        return v
    w = v.clone()
    u = v[idx].abs()
    sgn = v[idx].sign()
    w[idx] = sgn * proj_prob_simplex(u, r)
    return w


def proj_prob_simplex(v, r):
    """
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
    """
    dim = v.dim()
    assert dim <= 2

    mu = torch.sort(v, descending=True)[0]
    denom = torch.arange(1, v.size(-1) + 1, device=v.device).float()
    if dim > 1:
        denom.unsqueeze_(0)
    cumsum = (torch.cumsum(mu, dim=-1) - r) / denom
    neg_idx = (mu <= cumsum).nonzero(as_tuple=True)
    tmp = mu - cumsum
    tmp[neg_idx] = 1e9
    rho = tmp.argmin(-1)
    if dim == 1:
        theta = cumsum[rho]
        return F.relu(v - theta)
    theta = cumsum[(torch.arange(v.size(0)), rho)]
    return F.relu(v - theta.unsqueeze(1))


def proj_ball_intsct(inpt, center, norms, clip=None, max_iters=1000, tol=1e-5):
    """
    Project <input> to an intersection of norm balls specified by <norms>
    centered at  <center>.
    """

    # x has shape (num_norms, num_inputs, dim)
    if clip is not None:
        x = torch.cat([(inpt - center).unsqueeze(0)] * (len(norms) + 1), dim=0)
    else:
        x = torch.cat([(inpt - center).unsqueeze(0)] * len(norms), dim=0)
    u = torch.zeros_like(x)
    x_mean = x.mean(0)
    for _ in range(max_iters):
        for i, norm in enumerate(norms):
            x[i] = proj_ball(x_mean - u[i], 0, norm[1], norm[0])
        if clip is not None:
            x[-1] = torch.min(torch.max(x_mean - u[-1], clip[0] - center),
                              clip[1] - center)
        dual_res = x_mean - x.mean(0)
        prim_res = x - x_mean
        u += prim_res
        x_mean -= dual_res
        if (prim_res.norm(2, -1).max() < tol and dual_res.norm(2, -1).max() < tol):
            break
    return x_mean + center


def proj_polytope_batch(x_hat, A, AAT, norm_A, b, params, clip_center=None):
    """
    Project <x> onto a polytope defined by Ax <= b.
    For dual method, we assume that box constraint is incorporated in the
    polytope parameters (A, AAT, norm_A, b, b_hat).
    """
    if x_hat.dim() == 1:
        if (params['method'] in ['dual_ascent', 'gca', 'parallel_gca']
                and params['clip'] is not None and clip_center is not None):
            b_clip = b.clone()
            b_clip[- params['dim']:] -= clip_center
            b_clip[- 2 * params['dim']:- params['dim']] += clip_center
            x = proj_polytope(x_hat, A, AAT, norm_A, b_clip, params)
        else:
            x = proj_polytope(x_hat, A, AAT, norm_A, b, params)
    else:
        x = x_hat.clone()
        if (params['method'] in ['dual_ascent', 'gca', 'parallel_gca']
                and params['clip'] is not None and clip_center is not None):
            b_clip = torch.cat([b.unsqueeze(0), ] * x_hat.size(0), dim=0)
            b_clip[:, - params['dim']:] += clip_center
            b_clip[:, - 2 * params['dim']:- params['dim']] -= clip_center
        for i in range(x_hat.size(0)):
            if params['clip'] is not None and clip_center is not None:
                if params['method'] in ['dual_ascent', 'gca', 'parallel_gca']:
                    x[i] = proj_polytope(x_hat[i], A, AAT, b_clip[i], params)
                elif params['method'] == 'dykstra':
                    x[i] = proj_polytope(x_hat[i], A, AAT, b, params,
                                         clip_center=clip_center[i])
            else:
                x[i] = proj_polytope(x_hat[i], A, AAT, b, params)
        #     if (params['method'] in ['dual_ascent', 'gca', 'parallel_gca']
        #             and params['clip'] is not None and clip_center is not None):
        #         p = mp.Process(target=proj_polytope,
        #                        args=(x_hat[i], A, AAT, norm_A, b_clip[i], params))
        #     else:
        #         p = mp.Process(target=proj_polytope,
        #                        args=(x_hat[i], A, AAT, norm_A, b, params))
        #     p.start()
        #     processes.append(p)
        # for p in processes:
        #     p.join()
    return x


def proj_polytope(x_hat, A, AAT, b, params, idx_plane=None, clip_center=None):
    """
    Project <x> onto a polytope defined by Ax <= b.
    For dual method, we assume that box constraint is incorporated in the
    polytope parameters (A, AAT, norm_A, b, b_hat).
    """

    if params['method'] == 'dual_ascent':
        x = dual_ascent(x_hat, A, AAT, b, params)
    elif params['method'] == 'gca':
        x = gca(x_hat, A, AAT, b, params, idx_plane=idx_plane)
    elif params['method'] == 'parallel_gca':
        x = parallel_gca(x_hat, A, AAT, b, params)
    elif params['method'] == 'dykstra':
        x = admm_polytope(x_hat, A, b, params, clip_center=clip_center)
        # x = test(x_hat, A, norm_A, b, params)
    elif params['method'] == 'cvx':
        x = solve_qp(x_hat, A, b, params, x_init=None)
    else:
        raise NotImplementedError('method no implemented.')
    return x


def admm_polytope(x_hat, A, b, params, clip_center=None):
    """
    Find projection to a polytope specified by Ax <= b using parallel ADMM or
    Dykstra's algorithm.

    norm_A is a square of l2-norm of rows of A. If possible, try to normalize
    A first and pass 1 as norm_A to save compuation.
    """

    # if params['clip'] is not None:
    #     x = torch.cat([x_hat.unsqueeze(0)] * (A.size(0) + 1), dim=0)
    # else:
    #     x = torch.cat([x_hat.unsqueeze(0)] * A.size(0), dim=0)
    if params['clip'] is not None:
        u = torch.zeros((A.size(0) + 1, A.size(1)),
                        device=params['device'],
                        dtype=params['dtype'])
    else:
        u = torch.zeros_like(A)
    x_mean = x_hat.clone()

    for step in range(params['max_proj_iters']):
        # update x_i
        x = x_mean - u
        if params['clip'] is not None:
            res = ((A * x[:-1]).sum(1) - b)
            u[:-1] -= F.relu(res).unsqueeze(1) * A
            clipped = torch.min(torch.max(params['clip'][0] - clip_center, x[-1]),
                                params['clip'][1] - clip_center)
            u[-1] = clipped - x[-1]
        else:
            res = ((A * x).sum(1) - b)
            # idx = (res > 0).nonzero(as_tuple=True)
            # x[idx] -= res[idx].unsqueeze(1) * A[idx]
            u = - F.relu(res).unsqueeze(1) * A
            # x -= F.relu(res).unsqueeze(1) * A
        x += u
        # update u_i
        x_mean = x.mean(0)

        if params['early_stop']:
            if u.abs().max() < params['tol']:
                break

        # dual_res = x_mean - x.mean(0)
        # prim_res = x - x_mean
        # u += prim_res
        # x_mean -= dual_res

        # stopping criterion
        # if params['early_stop']:
        #     prim_tol = prim_res.norm(2, -1).max() < params['tol']
        #     dual_tol = dual_res.norm(2, -1).max() < params['tol']
        #     if prim_tol and dual_tol:
        #         break

    return x_mean


def test(x_hat, A, norm_A, b, params):

    # if params['clip'] is not None:
    #     z = torch.zeros((A.size(0) + 1, A.size(1)),
    #                     device=params['device'],
    #                     dtype=params['dtype'])
    # else:
    #     z = torch.zeros_like(A)
    z = torch.zeros(A.size(0), device=params['device'], dtype=params['dtype'])
    m = A.size(0)
    alpha = 1
    x = x_hat.clone()

    for step in range(params['max_proj_iters']):
        # update x_i
        # TODO: assume norm_A is 1
        c = torch.min(- (A * x).sum(1) + b, m * z)
        x += alpha * (c.unsqueeze(1) * A).mean(0)
        z -= c / m

    return x
