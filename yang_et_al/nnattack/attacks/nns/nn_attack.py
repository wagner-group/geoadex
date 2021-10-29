"""
Module for implementation of Region Based Attack
"""
import logging
from ..utils import solve_lp, solve_qp
from .cutils import c_get_half_space, get_all_half_spaces, get_constraints, check_feasibility
from tqdm import tqdm
from joblib import Parallel, delayed
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KDTree
import numpy as np
import cvxopt.glpk
from cvxopt import matrix, solvers
import itertools
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#solvers.options['solver'] = 'glpk'
#solvers.options['maxiters'] = 30
solvers.options['show_progress'] = False
#solvers.options['feastol'] = 1e-7
#solvers.options['abstol'] = 1e-7
#solvers.options['reltol'] = 1e-7
cvxopt.glpk.options["msg_lev"] = "GLP_MSG_OFF"

CONSTRAINTTOL = 5e-6


def get_half_space(a, b):
    w = (b - a)
    c = np.dot(w.T, (a + b) / 2)
    sign = -np.sign(np.dot(w.T, b) - c)
    w = sign * w
    c = sign * c
    return [w, c]

# @profile


def get_sol(target_x, tuple_x, faropp, kdtree,
            glob_trnX, glob_trny, init_x=None, n_jobs=1):
    tuple_x = np.asarray(tuple_x)
    trnX = np.copy(glob_trnX)
    emb_tar = target_x
    G, h, _ = get_constraints(trnX, tuple_x, kdtree, faropp, emb_tar)
    G, h = matrix(G, tc='d'), matrix(h, tc='d')

    #assert (transformer.shape[1] == glob_trnX.shape[1])
    n_fets = target_x.shape[0]

    Q = 2 * matrix(np.eye(n_fets), tc='d')
    #T = matrix(transformer.astype(np.float64), tc='d')

    #G = G * T
    q = matrix(-2*target_x, tc='d')

    temph = h - CONSTRAINTTOL  # make sure all constraints are met

    status, sol = solve_qp(np.array(Q), np.array(q), np.array(G),
                           np.array(temph), n_fets)
    if status == 'optimal':
        ret = sol.reshape(-1)
        return True, ret
    else:
        return False, None


def sol_sat_constraints(G, h) -> bool:
    """ Check if the constraint is satisfiable
    """
    fet_dim = G.shape[1]
    c = matrix(np.zeros(fet_dim), tc='d')
    G = matrix(G, tc='d')
    temph = matrix(h - CONSTRAINTTOL, tc='d')
    sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')
    return (sol['status'] == 'optimal')


def get_sol_l1(target_x, tuple_x, faropp, kdtree, glob_trnX,
               glob_trny, init_x=None):
    tuple_x = np.asarray(tuple_x)
    fet_dim = target_x.shape[0]

    emb_tar = target_x
    trnX = np.copy(glob_trnX)
    G, h, dist = get_constraints(trnX, tuple_x, kdtree, faropp, emb_tar)
    #G = np.dot(G, transformer)

    if init_x is None and not sol_sat_constraints(G, h):
        return False, None

    c = matrix(np.concatenate((np.zeros(fet_dim), np.ones(fet_dim))), tc='d')

    G = np.hstack((G, np.zeros((G.shape[0], fet_dim))))
    G = np.vstack((G, np.hstack((np.eye(fet_dim), -np.eye(fet_dim)))))
    G = np.vstack((G, np.hstack((-np.eye(fet_dim), -np.eye(fet_dim)))))

    h = np.concatenate((h, target_x, -target_x))

    G, h = matrix(G, tc='d'), matrix(h, tc='d')

    temph = h - CONSTRAINTTOL
    if init_x is not None:
        sol = solvers.lp(c=c, G=G, h=temph, solver='glpk',
                         initvals=init_x)
    else:
        sol = solvers.lp(c=c, G=G, h=temph, solver='glpk')

    if sol['status'] == 'optimal':
        ret = np.array(sol['x']).reshape(-1)
        return True, ret[:len(ret)//2]
    else:
        #logger.warning("solver error")
        return False, None

# @profile


def get_sol_linf(target_x, tuple_x, faropp, kdtree,
                 glob_trnX, glob_trny, init_x=None, n_jobs=1):
    tuple_x = np.asarray(tuple_x)
    fet_dim = target_x.shape[0]

    emb_tar = target_x
    trnX = np.copy(glob_trnX)
    G, h, _ = get_constraints(trnX, tuple_x, kdtree, faropp, emb_tar)
    #G = np.dot(G, transformer)

    if init_x is None and not sol_sat_constraints(G, h):
        return False, None

    c = np.concatenate((np.zeros(fet_dim), np.ones(1))).reshape((-1, 1))

    G2 = np.hstack((np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G3 = np.hstack((-np.eye(fet_dim), -np.ones((fet_dim, 1))))
    G = np.hstack((G, np.zeros((G.shape[0], 1))))
    G = np.vstack((G, G2, G3))
    h = np.concatenate((h, target_x, -target_x)).reshape((-1, 1))

    temph = h - CONSTRAINTTOL

    status, sol = solve_lp(c=c, G=G, h=temph, n=len(c), n_jobs=n_jobs)
    if status == 'optimal':
        ret = np.array(sol).reshape(-1)
        return True, ret[:-1]
    else:
        return False, None


def get_adv(target_x, target_y, kdtree, n_searches, n_neighbors, faropp,
            lp_sols, glob_trnX, glob_trny, ord=2, n_jobs=1):
    ind = kdtree.query(target_x.reshape((1, -1)),
                       k=n_neighbors, return_distance=False)[0]
    if target_y != np.argmax(np.bincount(glob_trny[ind])):
        # already incorrectly predicted
        return np.zeros_like(target_x)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(glob_trnX, glob_trny)
    pred_trny = knn.predict(glob_trnX)

    temp = (target_x, np.inf)
    if n_searches == -1:
        n_searches = glob_trnX.shape[0]
        ind = np.arange(glob_trnX.shape[0])
    else:
        ind = kdtree.query(target_x.reshape((1, -1)),
                           k=n_searches, return_distance=False)
        ind = ind[0]

    combs = []
    for comb in itertools.combinations(range(n_searches), n_neighbors):
        comb = list(comb)
        # majority
        if target_y != np.argmax(np.bincount(glob_trny[ind[comb]])):
            combs.append(comb)

    if ord == 1:
        get_sol_fn = get_sol_l1
    elif ord == 2:
        get_sol_fn = get_sol
    elif ord == np.inf:
        get_sol_fn = get_sol_linf
    else:
        raise ValueError("Unsupported ord %d" % ord)

    def _helper(comb, trnX, trny, init_x):
        comb_tup = tuple(ind[comb])
        ret, sol = get_sol_fn(target_x, ind[comb], faropp, kdtree,
                              trnX, trny, init_x=init_x, n_jobs=n_jobs)
        return ret, sol

    def not_vacum(x): return tuple(ind[x]) not in lp_sols or lp_sols[tuple(ind[x])]
    combs = list(filter(not_vacum, combs))
    if n_neighbors == 1:
        sols = Parallel(n_jobs=-1, verbose=1)(
            delayed(_helper)(comb, glob_trnX, glob_trny,
                             init_x=glob_trnX[ind[comb[0]]]) for comb in combs)
    else:
        sols = Parallel(n_jobs=-1, verbose=1)(
            delayed(_helper)(comb, glob_trnX, glob_trny, None) for comb in combs)
    status, sols = zip(*sols)
    sols = np.array(sols)
    for i, s in enumerate(status):
        if not s:
            assert sols[i] is None
            if n_neighbors == 1:
                # some time region is too small for solver
                sols[i] = glob_trnX[ind[combs[i]][0]]
                #lp_sols[tuple(ind[combs[i]])] = glob_trnX[ind[combs[i]][0]]
            else:
                lp_sols[tuple(ind[combs[i]])] = None

    #_, sols = list(zip(*list(filter(lambda s: True if s[0] else False, zip(status, sols)))))
    sols = np.array(list(filter(lambda x: np.linalg.norm(x) != 0, sols)))
    eps = np.linalg.norm(sols - target_x, axis=1, ord=ord)
    #temp = (sols[eps.argmin()], eps.min())
    return sols[eps.argmin()] - target_x


def attack_with_eps_constraint(perts, ord, eps):
    perts = np.asarray(perts)
    if isinstance(eps, list):
        rret = []
        norms = np.linalg.norm(perts, axis=1, ord=ord)
        for ep in eps:
            t = np.copy(perts)
            t[norms > ep, :] = 0
            rret.append(t)
        return rret
    elif eps is not None:
        perts[np.linalg.norm(perts, axis=1, ord=ord) > eps, :] = 0
        return perts
    else:
        return perts


def rev_get_adv(target_x, target_y, kdtree, n_searches, n_neighbors, faropp,
                lp_sols, glob_trnX, glob_trny, ord=2, method='self',
                knn: KNeighborsClassifier = None, n_jobs=1):
    if n_searches == -1:
        n_searches = glob_trnX.shape[0]
    temp = (target_x, np.inf)

    # already predicted wrong
    if knn.predict(target_x.reshape((1, -1)))[0] != target_y:
        return temp[0] - target_x

    if ord == 1:
        get_sol_fn = get_sol_l1
    elif ord == 2:
        get_sol_fn = get_sol
    elif ord == np.inf:
        get_sol_fn = get_sol_linf
    else:
        raise ValueError("Unsupported ord %d" % ord)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(glob_trnX, glob_trny)
    pred_trny = knn.predict(glob_trnX)

    ind = kdtree.query(target_x.reshape((1, -1)),
                       k=len(glob_trnX), return_distance=False)[0]
    ind = list(filter(lambda x: pred_trny[x] != target_y, ind))[:n_searches]

    solsss = []
    for i in ind:
        if method == 'self':
            inds = [i]
        elif method == 'region':
            procedX = glob_trnX[i].reshape((1, -1))
            inds = kdtree.query(procedX, k=n_neighbors, return_distance=False)[0]
        inds = tuple([_ for _ in inds])

        ret, sol = get_sol_fn(target_x, inds, faropp, kdtree,
                              glob_trnX, glob_trny, init_x=glob_trnX[i], n_jobs=n_jobs)
        solsss.append(sol)

        if method == 'region':
            #assert ret
            if not ret:
                proc = np.array([glob_trnX[i]])
                sol = np.array(glob_trnX[i])
            else:
                proc = np.array([sol])

            # DEBUG: If fail, return inf instead of triggering assert
            # assert knn.predict(proc)[0] != target_y
            if knn.predict(proc)[0] == target_y:
                # return np.zeros_like(target_x) + np.inf
                continue

            eps = np.linalg.norm(sol - target_x, ord=ord)
            if eps < temp[1]:
                temp = (sol, eps)
        elif ret:  # method == 'self'
            proc = np.array([sol])
            if knn.predict(proc)[0] != target_y:
                eps = np.linalg.norm(sol - target_x, ord=ord)
                if eps < temp[1]:
                    temp = (sol, eps)
    solsss = np.asarray(solsss)

    return temp[0] - target_x


class NNOptAttack():
    def __init__(self, trnX, trny, n_neighbors=3, n_searches=-1, faropp=-1,
                 transformer=None, ord=2, n_jobs=1):
        #furthest >= K
        self.n_jobs = n_jobs
        self.K = n_neighbors
        self.trnX = trnX
        self.trny = trny
        self.n_searches = min(n_searches, len(trnX))
        self.faropp = faropp
        self.transformer = transformer
        self.ord = ord
        if transformer is not None:
            self.tree = KDTree(self.transformer.transform(self.trnX))
        else:
            self.tree = KDTree(self.trnX)
        self.lp_sols = {}

    def perturb(self, X, y, eps=None, logging=False, n_jobs=1):
        raise NotImplementedError()


class NNAttack(NNOptAttack):
    def __init__(self, trnX, trny, n_neighbors=3, n_searches=-1, faropp=-1,
                 transformer=None, ord=2, n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                         n_searches=n_searches, faropp=faropp, transformer=transformer,
                         ord=ord, n_jobs=n_jobs)

    # @profile
    def perturb(self, X, y, eps=None, n_jobs=1):
        if self.transformer:
            transformer = self.transformer.transformer()
        else:
            transformer = np.eye(self.trnX.shape[1])

        #global glob_trnX
        #global glob_trny
        glob_trnX = self.trnX
        glob_trny = self.trny

        ret = []
        for i, (target_x, target_y) in tqdm(enumerate(zip(X, y)), ascii=True, desc="Perturb"):
            ret.append(get_adv(target_x.astype(np.float64), target_y, self.tree,
                               self.n_searches, self.K, self.faropp,
                               self.lp_sols,
                               glob_trnX=glob_trnX,
                               glob_trny=glob_trny,
                               ord=self.ord))

        self.perts = np.asarray(ret)
        return attack_with_eps_constraint(self.perts, self.ord, eps)


class KNNRegionBasedAttackExact(NNAttack):
    """
    Exact Region Based Attack (RBA-Exact) for K-NN

    Arguments:
        trnX {ndarray, shape=(n_samples, n_features)} -- Training data
        trny {ndarray, shape=(n_samples)} -- Training label

    Keyword Arguments:
        n_neighbors {int} -- Number of neighbors for the target k-NN classifier (default: {3})
        n_searches {int} -- Number of regions to search, -1 means all regions (default: {-1})
        ord {int} -- Order of the norm for perturbation distance, see numpy.linalg.norm for more information (default: {2})
        n_jobs {int} -- number of cores to run (default: {1})
    """

    def __init__(self, trnX, trny, n_neighbors=3, n_searches=-1, ord=2, n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                         n_searches=-1, faropp=-1, transformer=None, ord=ord, n_jobs=n_jobs)


class RevNNAttack(NNOptAttack):
    """
    Approximated Region Based Attack (RBA-Approx)

    Arguments:
        trnX {ndarray, shape=(n_samples, n_features)} -- Training data
        trny {ndarray, shape=(n_samples)} -- Training label

    Keyword Arguments:
        n_neighbors {int} -- Number of neighbors for the target k-NN classifier (default: {3})
        n_searches {int} -- Number of regions to search (default: {-1})
        ord {int} -- Order of the norm for perturbation distance, see numpy.linalg.norm for more information (default: {2})
        n_jobs {int} -- number of cores to run (default: {1})
        faropp {int} -- Not used (default: {-1})
        transformer {[type]} -- Not used (default: {None})
        method {str} -- Not used (default: {'region'})
    """

    def __init__(self, trnX: np.array, trny: np.array, n_neighbors: int = 3,
                 n_searches: int = -1, faropp: int = -1, transformer=None, ord=2,
                 method='region', n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                         n_searches=n_searches, faropp=faropp, transformer=transformer,
                         ord=ord)
        self.method = method

    # @profile
    def perturb(self, X, y, eps=None, n_jobs=1):
        if self.transformer:
            transformer = self.transformer.transformer()
        else:
            transformer = np.eye(self.trnX.shape[1])

        #global glob_trnX
        #global glob_trny
        glob_trnX = self.trnX
        glob_trny = self.trny

        knn = KNeighborsClassifier(n_neighbors=self.K)
        knn.fit(glob_trnX, glob_trny)
        X = X.astype(np.float64)

        n_jobs = -1
        if n_jobs == 1:
            ret = []
            for i, (target_x, target_y) in tqdm(enumerate(zip(X, y)), ascii=True, desc="Perturb"):
                ret.append(
                    rev_get_adv(target_x.astype(np.float64), target_y,
                                self.tree, self.n_searches, self.K, self.faropp,
                                self.lp_sols, ord=self.ord,
                                method=self.method, knn=knn, n_jobs=self.n_jobs,
                                glob_trnX=glob_trnX, glob_trny=glob_trny,
                                )
                )
        else:

            def _helper(target_x, target_y):
                return rev_get_adv(target_x, target_y,
                                   self.tree, self.n_searches, self.K, self.faropp,
                                   dict(), ord=self.ord,
                                   method=self.method, knn=knn, n_jobs=1,
                                   glob_trnX=glob_trnX, glob_trny=glob_trny,
                                   )
            ret = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(_helper)(tar_x, tar_y)
                for (tar_x, tar_y) in zip(X, y))

        self.perts = np.asarray(ret)
        return attack_with_eps_constraint(self.perts, self.ord, eps)


class KNNRegionBasedAttackApprox(RevNNAttack):
    """
    Approximated Region Based Attack (RBA-Approx) for K-NN

    Arguments:
        trnX {ndarray, shape=(n_samples, n_features)} -- Training data
        trny {ndarray, shape=(n_samples)} -- Training label

    Keyword Arguments:
        n_neighbors {int} -- Number of neighbors for the target k-NN classifier (default: {3})
        n_searches {int} -- Number of regions to search, -1 means all regions (default: {-1})
        ord {int} -- Order of the norm for perturbation distance, see numpy.linalg.norm for more information (default: {2})
        n_jobs {int} -- number of cores to run (default: {1})
    """

    def __init__(self, trnX: np.array, trny: np.array, n_neighbors: int = 3,
                 n_searches: int = -1, ord=2, n_jobs=1):
        super().__init__(trnX=trnX, trny=trny, n_neighbors=n_neighbors,
                         n_searches=n_searches, faropp=-1, transformer=None, ord=ord,
                         method='region', n_jobs=n_jobs)
