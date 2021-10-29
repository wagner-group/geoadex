import multiprocessing
import time

import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

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
    TOL = 1e-6

    def __init__(self, points, labels, k, log, compute_lb_dist=True):
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
        self.knn = NearestNeighbors(
            n_neighbors=k, algorithm='auto', leaf_size=30, n_jobs=None)
        self.knn.fit(points)

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
        Get a bisector between <point1> and <point2> with normal vector
        pointing towards <point1>: w[:-1] @ x + w[-1] = 0.
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

    def dist_to_facet(self, cur_cell, label, idx, facet, lb_dist, proj_query,
                      computed_cells, ub_facet, query, A, AAT, b, k, params):
        # skip if neighbor is already visited
        neighbor = self._get_neighbor_from_facet(cur_cell, facet)
        if neighbor in computed_cells:
            return None

        # check if facet is adv
        neighbor_label = self.classify(neighbor)
        is_adv = int(neighbor_label != label)

        if lb_dist[idx] > ub_facet[0]:
            # ignore facet further than ub_facet
            return None
        # NOTE: skip keeping tracking of lb for now. This may add more
        # computation than it saves.
        # if lb_dist[i] > lb_facet[0]:
        #     is_lb = 1
        #     # NOTE: be careful here
        #     # check if lb_dist is also true dist
        #
        #     # option 1: check by neighbors
        #     # knn, dists = self.get_knn(
        #     #     proj_query[i], k=k + 1, return_distance=True)
        #     # cond = set(neighbor).union(set(cur_cell)) == set(knn)
        #     # if cond:
        #     #     dist0 = dists[knn.index(facet[0])]
        #     #     dist1 = dists[knn.index(facet[1])]
        #     #     if abs(dist0 - dist1) < self.TOL:
        #     #         is_lb = 0
        #
        #     # option 2: check polytope contraints
        #     if self._check_point_in_polytope(proj_query[i], hplanes):
        #         is_lb = 0
        #
        #     # facet_list.append((lb_dist[i], (cur_cell, facet, 1)))
        #     Q.insert(lb_dist[i], (cur_cell, facet, is_adv, is_lb))

        # get k + 1 nearest neighbor of the naively projected point
        knn = self.get_knn(proj_query[idx], k=k + 1, return_distance=False)
        cond = set(neighbor).union(set(cur_cell)) == set(knn.flatten())
        if cond:
            return (lb_dist[idx], (neighbor, facet, is_adv, 0))

        # if cur_cell[0] == 53843 and neighbor[0] == 42:
        #     params['stop'] = True
        # else:
        #     params['stop'] = False
        #     return None

        proj = proj_polytope(query, A, AAT, b, params, idx_plane=idx)
        # skip if solver exits with None, either facet is further
        # than ub or this is an invalid facet
        if proj is None:
            return None
        print('good')

        # compute distance to neighbor cell directly
        start = time.time()
        _, nb_hplanes = self.get_neighbor_facet(neighbor, params)
        nb_A = nb_hplanes[:, :-1]
        nb_b = nb_hplanes[:, -1]
        nb_AAT = nb_A @ nb_A.T
        print(time.time() - start)
        proj = proj_polytope(query, nb_A, nb_AAT, nb_b, params, idx_plane=None)
        print(time.time() - start)
        # assert proj is not None, 'something is wrong here.'
        if proj is None:
            import pdb
            pdb.set_trace()

        true_dist = np.sum((query - proj) ** 2)
        # set lb bit to 0 since we compute true distance
        return (true_dist, (neighbor, facet, is_adv, 0))

    def get_neighbor_facet(self, cell, params):
        """Return known facets of <cell>."""
        # determine 1st-order neighbors of each point in cell if not done so
        # for point in cell:
        #     self._get_1st_order_neighbors(point, exact)

        # TODO: after approximate method
        # # get union of neighbors
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
        # TODO: use another method to compute upper bound quickly
        ub_facet = (np.inf, ([], (-1, -1), 0, 0))

        # get current cell of query: cur_cell
        cur_cell = self.knn.kneighbors(
            query.reshape(1, -1), k, return_distance=False)[0]
        cur_cell = tuple(sorted(cur_cell))
        # define a set of visited cells
        visited_cells = set({cur_cell})
        computed_cells = set({cur_cell})

        clabel = self.classify(cur_cell)
        if clabel != label:
            print('Finished: query is already misclassified.')
            return query

        # =================================================================== #
        num_cores = multiprocessing.cpu_count()
        with Parallel(n_jobs=num_cores) as parallel:
            while True:
                print('current cell: %s' % cur_cell)
                # print(self.points[cur_cell[0]])
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

                # if cur_cell == (639, ):
                #     import pdb
                #     pdb.set_trace()

                start = time.time()

                if not params['parallel']:
                    # sequential version
                    facets_list = []
                    for idx, facet in enumerate(facets):
                        if idx % 100 == 0:
                            print(idx)
                        facets_list.append(self.dist_to_facet(
                            cur_cell, label, idx, facet, lb_dist, proj_query,
                            computed_cells, ub_facet, query, A, AAT, b, k,
                            params))
                else:
                    # parallel version
                    facets_list = parallel(delayed(self.dist_to_facet)(
                        cur_cell, label, idx, facet, lb_dist, proj_query,
                        computed_cells, ub_facet, query, A, AAT, b, k,
                        params) for idx, facet in enumerate(facets))

                print('facet time: ', time.time() - start)
                start = time.time()

                for facet in facets_list:
                    if facet is not None:
                        computed_cells.add(facet[1][0])
                        Q.insert(facet[0], facet[1])
                        if facet[1][2] and facet[0] < ub_facet[0]:
                            ub_facet = facet

                print('PQ time: ', time.time() - start)

                # keep popping smallest item in Q until unvisited cell is found
                while True:
                    (dist, (cell, facet, is_adv, is_lb)) = Q.popitem()
                    neighbor = self._get_neighbor_from_facet(cell, facet)
                    if neighbor not in visited_cells:
                        visited_cells.add(
                            self._get_neighbor_from_facet(cell, facet))
                        break

                print(dist, (cell, facet, is_adv, is_lb))
                # print(Q)
                # print(visited_cells)
                # import pdb
                # pdb.set_trace()

                # if is_lb:
                #     # TODO: compute exact dist and update Q
                #     true_dist = self._recompute_dist(query, cell, facet)
                #     Q.insert(true_dist, (cell, facet, is_adv, 0))
                #     # update other facets in Q that are lb and closer than the new
                #     # true_dist
                #     for entry in Q.search_below(true_dist):
                #         # skip if already not lb
                #         if not entry[1][-1]:
                #             continue
                #         dist = self._recompute_dist(
                #             query, entry[1][0], entry[1][1])
                #         cell, facet, is_adv, _ = Q.pop(entry[0])
                #         assert cell == entry[1][0] and facet == entry[1][1]
                #         Q.insert(dist, (cell, facet, is_adv, 0))
                #     # pop the new item which should no longer be lb
                #     (dist, (cell, facet, is_adv, is_lb)) = Q.popitem()
                if is_adv:
                    # if it is adv then we are done
                    print('Finished: an adversarial cell is found!')
                    print('Number of cells visited: ', len(visited_cells))
                    break
                # set new lb_facet
                lb_facet = (dist, (cell, facet, is_adv, is_lb))
                # if not adv, look at the neighbor of the popped cell
                cur_cell = self._get_neighbor_from_facet(cell, facet)

        # =================================================================== #

        # TODO: recompute dist in approximate case
        return dist, cell, facet

    def get_knn(self, query, k=None, return_distance=False):
        """Get k-nearest neighbors of query."""
        if query.ndim == 1:
            queries = [query]
        else:
            queries = query
        knn = self.knn.kneighbors(queries, k, return_distance=return_distance)
        if return_distance:
            return knn[1], [np.array(dist) ** 2 for dist in knn[0]]
        return knn

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
