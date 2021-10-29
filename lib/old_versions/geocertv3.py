import multiprocessing
import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed, wrap_non_picklable_objects

from .myskiplist import MySkipList
from .projection import proj_polytope
from .solvers import solve_feas


class GeoCert:
    """
    - Distance is treated as squared Euclidean distance.
    - Cell is represented by a sorted tuple, even though we are not
    interested in the ordering.
    - Facet is a tuple with the first entry being a point in the current cell
    and the second being a point in the neighboring cell.
    """
    EPS = 1e-9
    TOL = 1e-7

    def __init__(self, points, labels, k, index, log, compute_lb_dist=True):
        self.points = points
        self.labels = labels
        self.k = k
        self.log = log
        self.num_points, self.dim = points.shape

        # table to store 1st-order neighboring relationship of data points.
        # Value meaning: 0: not neighbor, 1: neighbor, -1: unsure.
        # Diagonal entries indicate whether we've found "enough" neighbors for
        # given points (-1: no, 1: yes). "Enough" depends on approximate method
        # TODO:
        # self.nb_tab = np.zeros((self.num_points, self.num_points),
        #                        dtype=np.int8) - 1
        # self.nb_tab = np.zeros((self.num_points, self.num_points),
        #                        dtype=np.int8) + 1

        # dictionary (hashtable) to store cells and their facets
        # (key: cell, value: list of facets)
        # self.facet_dict = {}

        # undirected graph to store neighboring relationship of kth-order cells
        self.G = nx.Graph()

        # create index for kNN search, data structure is automatically
        # determined by sklearn
        # self.knn = NearestNeighbors(
        #     n_neighbors=k, algorithm='auto', leaf_size=30, n_jobs=None)
        # self.knn.fit(points)
        self.knn = index

    # def _get_1st_order_neighbors(self, point, exact):
    #     """Get 1st-order neighbor of a given <point>."""
    #     # TODO: consider disgarding this method
    #     # NOTE: can use another way to check if we have found "enough"
    #     # neighbors for a give point
    #     if self.nb_tab[point, point] == 1:
    #         return
    #
    #     if exact:
    #         # list all other data points and compute hyperplanes
    #         others = np.arange(self.num_points, dtype=np.int64)
    #         others = others[others != point]
    #         hplanes = np.zeros((self.num_points - 1, self.dim + 1))
    #         self._get_hyperplane(point, others, hplanes)
    #
    #         # NOTE: facet further than ub dist can also be ignored here if
    #         # we do not care about building nb_tab, e.g. only have one query
    #         mask = np.ones(len(hplanes), dtype=np.bool)
    #         for i, other_point in enumerate(others):
    #             if self.check_facet([point], hplanes, i, mask=mask):
    #                 self._update_neighbor(point, other_point, 1)
    #             else:
    #                 # mask out some hyperplanes that are not facets
    #                 mask[i] = False
    #                 self._update_neighbor(point, other_point, 0)
    #         self.nb_tab[point, point] = 1
    #     else:
    #         # TODO:
    #         raise NotImplementedError()

    def _get_facets_from_cell(self, cell):
        """List all known facets of <cell>."""
        neighbors = self.G.neighbors(cell)
        facets = []
        for neighbor in neighbors:
            facets.append(self._get_facet_from_neighbor(cell, neighbor))
        return facets

    def _get_hyperplane(self, point1, point2, w):
        """
        Get a bisector between <point1> and <point2> that defines a halfspace
        covering <point1>.: w[:-1] @ x + w[-1] = 0.
        Update <w> in-place.
        """
        point1 = self.points[point1]
        point2 = self.points[point2]
        midpoint = (point1 + point2) / 2

        # normal vector is normalized to have norm of 1
        diff = point2 - point1
        if point1.ndim == point2.ndim:
            w[:-1] = diff / np.maximum(np.linalg.norm(diff, 2), self.EPS)
            w[-1] = w[:-1] @ midpoint
        else:
            w[:, :-1] = diff / \
                np.maximum(np.linalg.norm(diff, 2, 1), self.EPS)[:, np.newaxis]
            w[:, -1] = (midpoint * w[:, :-1]).sum(1)

    # def _recompute_dist(self, query, cell, facet):
    #     """Recompute distance from <query> to <facet> in a given <cell>."""
    #     facets = self._get_facets_from_cell(cell)
    #     hplanes = np.zeros((len(facets), self.dim + 1))
    #     for i, fct in enumerate(facets):
    #         # TODO: check if in-place update works
    #         self._get_hyperplane(fct[0], fct[1], hplanes[i])
    #     return proj_polytope(query, A, AAT, b, params, idx_plane=facets.index(facet))

    def _update_graph(self, cell, facets):
        """Update <self.G> with the new edges <facets> on node <cell>."""
        self.G.add_node(cell)
        for facet in facets:
            neighbor = self._get_neighbor_from_facet(cell, facet)
            self.G.add_edge(cell, neighbor)

    # def _update_neighbor(self, nb1, nb2, val):
    #     """update neighboring relationship of points"""
    #     self.nb_tab[nb1, nb2] = val
    #     self.nb_tab[nb2, nb1] = val

    def check_facet(self, cell, hyperplanes, idx, mask=None):
        """
        Check if a hyperplane at index <idx> of <hyperplanes> is an facet
        of <cell>.
        """
        # TODO
        return solve_feas(hyperplanes[mask], idx)

    def classify(self, cell):
        """get majority label of <cell>."""
        return np.bincount(self.labels[list(cell)]).argmax()

    @staticmethod
    def dist_to_facet(cur_cell, idx, facet, query, A, AAT, b, params, is_adv):

        # start = time.time()
        proj = proj_polytope(query, A, AAT, b, params, idx_plane=idx)
        # skip if solver exits with None, either facet is further
        # than ub or this is an invalid facet
        if proj is None:
            return None
        # print('solve time: ', time.time() - start)

        true_dist = np.sum((query - proj) ** 2)
        # set lb bit to 0 since we compute true distance
        return (true_dist, (cur_cell, facet, is_adv, 0))

    def get_neighbor_facet(self, cell, params):
        """Return known facets of <cell>."""
        # determine 1st-order neighbors of each point in cell if not done so
        # for point in cell:
        #     self._get_1st_order_neighbors(point, exact)

        # get union of neighbors
        # union_nb = set()
        # for point in cell:
        #     neighbor = np.where(self.nb_tab[point] == 1)[0]
        #     union_nb.update(neighbor)
        union_nb = set(range(self.points.shape[0]))
        # subtract elements in cell from union_nb
        union_nb = list(union_nb - set(cell))
        len_nb = len(union_nb)

        # create a list of all possible hyperplanes
        hplanes = np.zeros((len_nb * len(cell), self.dim + 1),
                           dtype=params['dtype'])
        facets = []
        for i, point in enumerate(cell):
            # TODO: check if in-place update works
            self._get_hyperplane(
                point, union_nb, hplanes[i * len_nb:(i + 1) * len_nb])
            for other in union_nb:
                facets.append((point, other))

        # TODO: most of this part can be skipped if we combine check_facet and
        # distance computation
        # # list known facets in cell
        # known_facets = self._get_facets_from_cell(cell)
        #
        # # NOTE: compute lb dist to each facet first and if it's larger than ub
        # # then it can be discarded
        #
        # # check whether each hyperplane is a facet
        # # NOTE: this step may be combined with distance computation
        # idx_facets = []
        # mask = np.ones(len(hplanes), dtype=np.bool)
        # for i, _ in enumerate(hplanes):
        #     if facets[i] in known_facets:
        #         idx_facets.append(i)
        #         continue
        #     if self.check_facet(cell, hplanes, i, mask=mask):
        #         # TODO: consider approximate case where neighbor facet might
        #         # not actuall exist
        #         # if not exact:
        #         idx_facets.append(i)
        #     else:
        #         mask[i] = False
        #
        # true_facets = facets[idx_facets]
        # # facet_val = self.facet_dict.get(cell)
        # # if facet_val is not None:
        # #     facet_val.extend(true_facets)
        # # else:
        # #     self.facet_dict[cell] = true_facets
        # # update graph
        # self._update_graph(cell, true_facets)
        #
        # return true_facets, hplanes[idx_facets]

        return facets, hplanes

    def get_cert(self, query, label, params, k=None):
        """
        Compute adv certificate of <query>.
        """
        if k is None:
            k = self.k

        # create skip list to store facets described by
        # (key: distance, value: (cell, facet, is_adv, is_lb))
        Q = MySkipList()
        # the most recent facet popped
        lb_facet = (np.inf, ([], (-1, -1), 0, 0))
        # the closest adv facet found so far
        ub_facet = (params['upperbound'], ([], (-1, -1), 0, 0))

        # get current cell of query: cur_cell
        _, indices = self.knn.search(query.reshape(1, -1), k)
        cur_cell = tuple(sorted(indices[0]))
        # define a set of visited cells
        visited_cells = set({cur_cell})

        clabel = self.classify(cur_cell)
        if clabel != label:
            print('Finished: query is already misclassified.')
            return query

        # XXT = self.points @ self.points.T

        # num_cores = multiprocessing.cpu_count()
        with Parallel(n_jobs=params['num_cores']) as parallel:
            while True:
                print('current cell: %s' % cur_cell)
                # determine neighboring facets of cur_cell
                facets, hplanes = self.get_neighbor_facet(cur_cell, params)

                # compute lb distance to facets, compute true distance if lb is
                # smaller than distance to lb_facet
                lb_dist, proj_query = self._compute_lb_dist(
                    query, hplanes, return_proj=True)

                # extract A, AAT, and b from hplanes
                # TODO: handle box constraint
                A = hplanes[:, :-1]
                b = hplanes[:, -1]
                AAT = A @ A.T

                # import pdb
                # pdb.set_trace()
                #
                # x = self.points[cur_cell[0]]
                # Xx = self.points @ x
                # tmp = XXT - Xx - Xx.T + x @ x
                #
                # import pdb
                # pdb.set_trace()

                # sort facets in descending order of lower bound distance
                indices = lb_dist.argsort()
                # ignore facet further than ub_facet
                indices = indices[lb_dist[indices] < ub_facet[0]]
                print('num facets filtered by projection: ',
                      len(facets) - len(indices))

                # check if there is any adv facet
                adv_indices, benign_indices = [], []
                adv_nb, benign_nb = [], []
                for idx in indices:
                    # find the neighboring cell
                    neighbor = self._get_neighbor_from_facet(
                        cur_cell, facets[idx])
                    if neighbor in visited_cells:
                        continue
                    # check label of neighbor
                    neighbor_label = self.classify(neighbor)
                    if neighbor_label != label:
                        adv_indices.append(idx)
                        adv_nb.append(neighbor)
                    else:
                        benign_indices.append(idx)
                        benign_nb.append(neighbor)

                print('num adv facets: %d, num benign facets: %d' % (
                    len(adv_indices), len(benign_indices)))

                start = time.time()

                # compute distance to adv facet first to set good upper bound
                for idx, neighbor in zip(adv_indices, adv_nb):
                    if lb_dist[idx] > ub_facet[0]:
                        break
                    facet = self.dist_to_facet(cur_cell, idx, facets[idx],
                                               query, A, AAT, b, params, 1)
                    if facet is None:
                        continue
                    if facet[0] >= ub_facet[0]:
                        continue
                    Q.insert(facet[0], facet[1])
                    ub_facet = facet
                    params['upperbound'] = facet[0]
                    print('new ub_facet with distance %.4f is set.' % facet[0])
                print('Done with adv facets.')

                # filter benign facets closer than ub_facet
                mask = np.where(lb_dist[benign_indices] < ub_facet[0])[0]
                benign_indices = np.array(benign_indices)[mask]
                benign_nb = np.array(benign_nb)[mask]

                # # get k + 1 nearest neighbor of the naively projected point
                # knn = self.get_knn(proj_query, k=k + 1, return_distance=False)
                # cond = set(neighbor).union(set(cur_cell)) == set(knn.flatten())
                # if cond:
                #     return (lb_dist, (cur_cell, facet, is_adv, 0))

                # we can now parallel computation on benign facets because
                # we are no longer updating up_facet
                if params['parallel']:
                    facets_list = parallel(delayed(self.dist_to_facet)(
                        cur_cell, idx, facets[idx], query, A, AAT, b,
                        params, 0) for idx, neighbor in
                        zip(benign_indices, benign_nb))
                    for facet in facets_list:
                        if facet is None:
                            continue
                        if facet[0] >= ub_facet[0]:
                            continue
                        Q.insert(facet[0], facet[1])
                else:
                    for idx, neighbor in zip(benign_indices, benign_nb):
                        # if lb_dist[idx] > ub_facet[0]:
                        #     break
                        facet = self.dist_to_facet(
                            cur_cell, idx, facets[idx], query, A, AAT, b,
                            params, 0)
                        if facet is None:
                            continue
                        if facet[0] >= ub_facet[0]:
                            continue
                        Q.insert(facet[0], facet[1])
                print('Done with benign facets.')

                print(time.time() - start)

                # keep popping smallest item in Q until an unvisited cell is found
                while True:
                    (dist, (cell, facet, is_adv, is_lb)) = Q.popitem()
                    neighbor = self._get_neighbor_from_facet(cell, facet)
                    if neighbor not in visited_cells:
                        visited_cells.add(
                            self._get_neighbor_from_facet(cell, facet))
                        break

                print(dist, (cell, facet, is_adv, is_lb))

                if is_adv:
                    # if it is adv then we are done
                    print('Finished: an adversarial cell is found!')
                    print('Number of cells visited: ', len(visited_cells))
                    break
                # set new lb_facet
                lb_facet = (dist, (cell, facet, is_adv, is_lb))
                # if not adv, look at the neighbor of the popped cell
                cur_cell = self._get_neighbor_from_facet(cell, facet)

        # TODO: recompute dist in approximate case
        return dist, cell, facet

    def get_knn(self, query, k=None, return_distance=False):
        """Get k-nearest neighbors of query."""
        if query.ndim == 1:
            # queries = [query]
            queries = query.reshape(1, -1)
        else:
            queries = query
        knn = self.knn.search(queries, k)
        # if return_distance:
        #     return ([np.array(dist) ** 2 for dist in knn[0]],
        #             [set(nn) for nn in knn[1]])
        # return [set(nn) for nn in knn]
        if return_distance:
            return knn[1], [np.array(dist) ** 2 for dist in knn[0]]
        return knn[1]

    @classmethod
    def _check_point_in_polytope(cls, point, hplanes):
        """
        Determine is <point> is inside all halfspaces represented by <hplanes>.
        """
        # TODO: change this to fast index search for k + 1 nearest neighbors
        return np.all(hplanes[:, :-1] @ point + hplanes[:, -1] >= - cls.TOL)

    @classmethod
    def _compute_lb_dist(cls, query, hyperplanes, return_proj=False):
        """Compute naive lb distance from <query> to all <hyperplanes>."""
        dist = hyperplanes[:, :-1] @ query - hyperplanes[:, -1]
        if return_proj:
            proj = query - dist.reshape(-1, 1) * hyperplanes[:, :-1]
            return dist ** 2, proj
        return dist ** 2

    @staticmethod
    def _get_facet_from_neighbor(cell, neighbor):
        """Get facet that borders <cell> and <neighbor>"""
        # NOTE: assume cell and neighbor are actually neighbors
        i = j = 0
        while True:
            if cell[i] != neighbor[j]:
                if cell[i] in neighbor:
                    facet2 = neighbor[j]
                    j += 1
                else:
                    facet1 = cell[i]
                    i += 1
            i += 1
            j += 1
            if i >= len(cell) or j >= len(cell):
                break
        return (facet1, facet2)

    @staticmethod
    def _get_neighbor_from_facet(cur_cell, facet):
        """Find a neighbor of <cur_cell> that shares same <facet>"""
        # NOTE: assume one of point in <facet> is in <cur_cell>
        # if facet[0] in cur_cell:
        #     neighbor = [facet[1] if x == facet[0] else x for x in cur_cell]
        # else:
        #     neighbor = [facet[0] if x == facet[1] else x for x in cur_cell]
        neighbor = [facet[1] if x == facet[0] else x for x in cur_cell]
        return tuple(sorted(neighbor))
