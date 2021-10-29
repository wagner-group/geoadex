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
        self.num_classes = len(np.unique(labels))
        self.k = k
        self.log = log
        self.num_points, self.dim = points.shape

        # dictionary to store neighboring relationship
        self.nb_dict = {}

        # undirected graph to store neighboring relationship of kth-order cells
        self.G = nx.Graph()

        # create index for kNN search, data structure is automatically
        # determined by sklearn
        # self.knn = NearestNeighbors(
        #     n_neighbors=k, algorithm='auto', leaf_size=30, n_jobs=None)
        # self.knn.fit(points)
        self.knn = index

        # compute class mean
        self.class_means = np.zeros((self.num_classes, self.dim),
                                    dtype=points.dtype)
        for lab in labels:
            self.class_means[lab] = points[labels == lab].mean(0)

    def compute_potential(self, query, label, proj, params):
        potential = np.linalg.norm(query - proj)
        if params['use_potential']:
            class_dists = np.sum((self.class_means - proj) ** 2, 1)
            true_class_dist = class_dists[label]
            class_dists[label] += 1e9
            other_class = class_dists.argmin()
            alpha = 0.5 / (np.linalg.norm(self.class_means[label]
                                          - self.class_means[other_class]))
            potential += alpha * (true_class_dist
                                  - class_dists[other_class])
        return potential

    def _get_1st_order_neighbors(self, point, params, mask=None, parallel=None):
        """Get 1st-order neighbor of a given <point>."""
        if mask is None:
            hplanes = np.zeros((self.num_points, self.dim + 1),
                               dtype=params['dtype'])
            self._get_hyperplane(point, self.points, hplanes)
        else:
            hplanes = np.zeros((len(mask), self.dim + 1),
                               dtype=params['dtype'])
            self._get_hyperplane(point, mask, hplanes)
        A = hplanes[:, :-1]
        b = hplanes[:, -1]
        AAT = A @ A.T

        # We check if a neighbor is valid by computing distance from the
        # current point to the bisesctor (facet) between them.
        neighbors = range(self.num_points) if mask is None else mask
        nb_list = []
        # We don't want to rule out with upper bound
        temp_upperbound = params['upperbound']
        temp_dist_choice = params['compute_dist_to_cell']
        params['upperbound'] = np.inf
        params['compute_dist_to_cell'] = False
        x_hat = self.points[point]

        if params['parallel'] and parallel is not None:
            facets_list = parallel(delayed(self.dist_to_facet)(
                None, idx, None, x_hat, None, A, AAT, b, params, 0)
                for idx, neighbor in enumerate(neighbors))
            for idx, neighbor in enumerate(neighbors):
                if facets_list[idx] is None or neighbor == point:
                    continue
                nb_list.append(neighbor)
        else:
            for idx, neighbor in enumerate(neighbors):
                if neighbor == point:
                    continue
                proj = self.dist_to_facet(
                    None, idx, None, x_hat, None, A, AAT, b, params, 0)
                if proj is None:
                    continue
                nb_list.append(neighbor)
        print('num neighbors of point %d: %d/%d' %
              (point, len(nb_list), len(neighbors)))
        params['upperbound'] = temp_upperbound
        params['compute_dist_to_cell'] = temp_dist_choice
        self.nb_dict[point] = nb_list

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

    def _get_precise_label(self, inpt, k):
        """
        Use this method to classify when <inpt> is close to or on multiple
        bisectors. Normal knn can be ambiguous in this case.
        """
        dist = np.sum((inpt - self.points) ** 2, 1)
        # find distance to the kth neighbor
        k_dist = np.sort(dist)[k - 1]
        indices = np.where(dist - k_dist < self.TOL)[0]
        close_indices = np.where(np.abs(dist - k_dist) < self.TOL)[0]
        sure_indices = np.setdiff1d(indices, close_indices)
        close_labels = self.labels[close_indices]
        sure_labels = self.labels[sure_indices]
        close_counts = np.bincount(close_labels, minlength=self.num_classes)
        sure_counts = np.bincount(sure_labels, minlength=self.num_classes)
        label_counts = sure_counts + \
            np.minimum(close_counts, k - len(sure_indices))
        # return np.where(label_counts == np.max(label_counts))[0]
        return np.where(label_counts >= np.ceil(k / 2))[0]

    def _update_graph(self, cell, facets):
        """Update <self.G> with the new edges <facets> on node <cell>."""
        self.G.add_node(cell)
        for facet in facets:
            neighbor = self._get_neighbor_from_facet(cell, facet)
            self.G.add_edge(cell, neighbor)

    def classify(self, cell):
        """get majority label of <cell>."""
        return np.bincount(self.labels[list(cell)]).argmax()

    # @staticmethod
    def dist_to_facet(self, cur_cell, idx, facet, query, label, A, AAT, b,
                      params, is_adv):

        if not params['compute_dist_to_cell']:
            # compute distance to facet
            proj = proj_polytope(query, A, AAT, b, params, idx_plane=idx)
        else:
            # compute distance to cell
            neighbor = self._get_neighbor_from_facet(cur_cell, facet)
            _, nb_hplanes = self.get_neighbor_facet(neighbor, params)
            nb_A = nb_hplanes[:, :-1]
            nb_b = nb_hplanes[:, -1]
            nb_AAT = nb_A @ nb_A.T
            proj = proj_polytope(
                query, nb_A, nb_AAT, nb_b, params, idx_plane=None)
        # skip if solver exits with None, either facet is further
        # than ub or this is an invalid facet
        if proj is None:
            return None

        true_dist = self.compute_potential(query, label, proj, params)
        # set lb bit to 0 since we compute true distance
        return (true_dist, (cur_cell, facet, is_adv, 0))

    def get_neighbor_facet(self, cell, params, parallel=None):
        """Return known facets of <cell>."""

        # generate set of potential neighbors to consider
        if params['neighbor_method'] == 'all':
            masks = [list(range(self.num_points)), ] * len(cell)
        elif params['neighbor_method'] == 'm_nearest':
            _, masks = self.knn.search(
                self.points[list(cell)].reshape(len(cell), -1),
                params['m'] + 1)
            # exclude the point itself from neighbor list
            masks = [mask[1:] for mask in masks]
        else:
            raise NotImplementedError('no specified approximate neighbors.')

        union_nb = set()
        if params['save_1nn_nb']:
            # find 1st-order neighbors and save them for future use
            for i, point in enumerate(cell):
                try:
                    union_nb.update(self.nb_dict[point])
                except KeyError:
                    print('computing 1st-order neighbor of point: ', point)
                    start = time.time()
                    self._get_1st_order_neighbors(
                        point, params, mask=masks[i], parallel=parallel)
                    union_nb.update(self.nb_dict[point])
                    print('neighbors saved. finish in %.2fs.' %
                          (time.time() - start))
        else:
            # If we do not plan to reuse 1st-order neighbors, then there is
            # no point specifying the true neighbors. Computing distance to the
            # kth-order cell will automatically remove unncessary facets.
            for mask in masks:
                union_nb.update(mask)

        # subtract elements in cell from union_nb
        union_nb = list(union_nb - set(cell))
        len_nb = len(union_nb)

        # create a list of all possible hyperplanes
        hplanes = np.zeros((len_nb * len(cell), self.dim + 1),
                           dtype=params['dtype'])
        facets = []
        for i, point in enumerate(cell):
            self._get_hyperplane(
                point, union_nb, hplanes[i * len_nb:(i + 1) * len_nb])
            for other in union_nb:
                facets.append((point, other))

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
        # define cells that GeoCert has visited
        visited_cells = set({cur_cell})
        # define cells that we have computed distance to
        computed_cells = set({cur_cell})

        clabel = self.classify(cur_cell)
        if clabel != label:
            print('Finished: query is already misclassified.')
            return query

        # XXT = self.points @ self.points.T

        # num_cores = multiprocessing.cpu_count()
        # with Parallel(n_jobs=params['num_cores']) as parallel:
        parallel = None
        start_main = time.time()

        # ==================== starting the main loop ======================= #

        while True:
            print('current cell: %s' % list(cur_cell))
            # determine neighboring facets of cur_cell
            facets, hplanes = self.get_neighbor_facet(
                cur_cell, params,
                parallel=parallel if params['parallel'] else None)

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

            # sort facets in descending order of lower bound distance
            indices = lb_dist.argsort()
            # ignore facet further than ub_facet
            indices = indices[lb_dist[indices] < ub_facet[0]]
            print('num facets filtered by projection: ',
                  len(facets) - len(indices))

            # create list of adv and benign facets that we will check
            adv_indices, benign_indices = [], []
            adv_nb, benign_nb = [], []
            for idx in indices:
                # find the neighboring cell
                neighbor = self._get_neighbor_from_facet(
                    cur_cell, facets[idx])
                if neighbor in visited_cells:
                    continue
                if ((params['treat_facet_as_cell']
                     or params['compute_dist_to_cell'])
                        and neighbor in computed_cells):
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
                facet = self.dist_to_facet(cur_cell, idx, facets[idx], query,
                                           label, A, AAT, b, params, 1)
                if facet is None:
                    continue
                if facet[0] >= ub_facet[0]:
                    continue
                Q.insert(facet[0], facet[1])
                computed_cells.add(neighbor)
                ub_facet = facet
                params['upperbound'] = facet[0]
                print('>>>>>>>>>>>>>>>>> time: %.2fs, new ub_facet with distance %.4f is set. <<<<<<<<<<<<<<<' %
                      (time.time() - start_main, facet[0]))
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
                    cur_cell, idx, facets[idx], query, label, A, AAT, b,
                    params, 0) for idx, neighbor in
                    zip(benign_indices, benign_nb))
                for i, facet in enumerate(facets_list):
                    if facet is None:
                        continue
                    if facet[0] >= ub_facet[0]:
                        continue
                    Q.insert(facet[0], facet[1])
                    computed_cells.add(tuple(benign_nb[i]))
            else:
                for idx, neighbor in zip(benign_indices, benign_nb):
                    # if lb_dist[idx] > ub_facet[0]:
                    #     break
                    # start = time.time()
                    facet = self.dist_to_facet(
                        cur_cell, idx, facets[idx], query, label, A, AAT, b,
                        params, 0)
                    # print(time.time() - start)
                    if facet is None:
                        continue
                    if facet[0] >= ub_facet[0]:
                        continue
                    Q.insert(facet[0], facet[1])
                    computed_cells.add(tuple(neighbor))
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
                print('num cells visited: %d, num cells computed: %d' %
                      (len(visited_cells), len(computed_cells)))
                break
            # set new lb_facet
            lb_facet = (dist, (cell, facet, is_adv, is_lb))
            # if not adv, look at the neighbor of the popped cell
            cur_cell = self._get_neighbor_from_facet(cell, facet)

        # ======================== end while =============================== #

        # Recompute distance to found adv cell
        adv_cell = self._get_neighbor_from_facet(cell, facet)
        params['neighbor_method'] = 'all'
        params['save_1nn_nb'] = False
        facets, hplanes = self.get_neighbor_facet(
            adv_cell, params,
            parallel=parallel if params['parallel'] else None)
        A = hplanes[:, :-1]
        b = hplanes[:, -1]
        AAT = A @ A.T
        params['max_proj_iters'] = 10000
        params['upperbound'] = np.inf
        proj = proj_polytope(query, A, AAT, b, params, idx_plane=None)
        final_label = self._get_precise_label(proj, k)
        is_adv = np.any(final_label != label)
        if not is_adv:
            # import pdb
            # pdb.set_trace()
            raise AssertionError('Obtain an invalid cell.')
        print('dist: %.4f' % np.linalg.norm(query - proj))
        return proj
        # return dist, cell, facet

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
