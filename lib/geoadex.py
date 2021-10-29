'''Multiprocess version'''
import time
from copy import deepcopy

import numpy as np
from joblib import Parallel, delayed

from .myskiplist import MySkipList
from .polytope_dual import gca


class GeoAdEx:
    """
    - Distance is treated as squared Euclidean distance.
    - Cell is represented by a sorted tuple, even though we are not
    interested in the ordering.
    - Facet is a tuple with the first entry being a point in the current cell
    and the second being a point in the neighboring cell.
    """
    EPS = 1e-9
    TOL = 1e-6
    # TOL = 1e-9
    # Define some exit codes
    FAIL = 0        # Fail to find an optimal adversarial examples
    SUCCESS = 1     # Optimal adversarial examples found
    TIMELIM = 2     # Maximum time limit is reached
    MISCLF = 3      # The query is already classified

    def __init__(self, points, labels, k, index, log, approx_index=None):
        self.points = points
        self.labels = labels
        self.num_classes = len(np.unique(labels))
        self.k = k
        self.log = log
        self.num_points, self.dim = points.shape

        # dictionary to store neighboring relationship of 1st-order cells
        self.nb_dict = {}

        # undirected graph to store neighboring relationship of kth-order cells
        # self.G = nx.Graph()
        # self.knb_dict = {}

        # index for kNN search
        self.knn = index
        self.approx_knn = approx_index if approx_index is not None else index

        # DEPRECATED: Compute class mean (only used with potential)
        self.class_means = np.zeros((self.num_classes, self.dim),
                                    dtype=points.dtype)
        for i in range(self.num_classes):
            self.class_means[i] = points[labels == i].mean(0)

    def compute_potential(self, query, label, proj, params):
        # DEPRECATED: potential doesn't work here
        potential = np.linalg.norm(query - proj)
        if params['use_potential']:
            raise NotImplementedError()
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
        # FIXME: we are not actually validating first-order neighbors
        self.nb_dict[point] = mask
        return

        # if mask is None:
        #     hplanes = np.zeros((self.num_points, self.dim + 1), dtype=params['dtype'])
        #     self._get_hyperplane(point, self.points, hplanes)
        # else:
        #     hplanes = np.zeros((len(mask), self.dim + 1), dtype=params['dtype'])
        #     self._get_hyperplane(point, mask, hplanes)
        # A = hplanes[:, :-1]
        # b = hplanes[:, -1]
        # AAT = A @ A.T

        # # We check if a neighbor is valid by computing distance from the
        # # current point to the bisesctor (facet) between them.
        # neighbors = range(self.num_points) if mask is None else mask
        # nb_list = []
        # x_hat = self.points[point]

        # if params['parallel'] and parallel is not None:
        #     facets_list = parallel(delayed(self.dist_to_facet)(
        #         None, idx, None, x_hat, None, A, AAT, b, params, 0, ub=np.inf,
        #         dist_to_cell=False) for idx, neighbor in enumerate(neighbors))
        #     for idx, neighbor in enumerate(neighbors):
        #         if facets_list[idx] is None or neighbor == point:
        #             continue
        #         nb_list.append(neighbor)
        # else:
        #     for idx, neighbor in enumerate(neighbors):
        #         if neighbor == point:
        #             continue
        #         proj = self.dist_to_facet(
        #             None, idx, None, x_hat, None, A, AAT, b, params, 0,
        #             ub=np.inf, dist_to_cell=False)
        #         if proj is None:
        #             continue
        #         nb_list.append(neighbor)
        # self.log.info(f'num neighbors of point {point}: {len(nb_list)}/{len(neighbors)}')
        # self.nb_dict[point] = nb_list

    def _get_hyperplane(self, point1, point2, w):
        """
        Get a bisector between <point1> and <point2> that defines a halfspace
        covering <point1>.: w[:-1] @ x + w[-1] <= 0.
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
        # Find distance to the kth neighbor
        k_dist = np.sort(dist)[k - 1]
        indices = np.where(dist - k_dist < self.TOL)[0]

        # counts = np.bincount(self.labels[indices], minlength=self.num_classes)
        # counts = np.minimum(counts, k)
        # return np.where(counts == counts.max())[0]

        close_indices = np.where(np.abs(dist - k_dist) < self.TOL)[0]
        sure_indices = np.setdiff1d(indices, close_indices)
        close_labels = self.labels[close_indices]
        sure_labels = self.labels[sure_indices]
        close_counts = np.bincount(close_labels, minlength=self.num_classes)
        sure_counts = np.bincount(sure_labels, minlength=self.num_classes)

        num_to_fill = k - sure_counts.sum()
        # If number of sure counts is k, then we are done
        assert num_to_fill >= 0
        if num_to_fill == 0:
            max_count = sure_counts.max()
            return np.where(sure_counts == max_count)[0]

        y_pred = []
        for i in range(self.num_classes):
            num_fill = min(num_to_fill, close_counts[i])
            new_counts = deepcopy(sure_counts)
            new_counts[i] += num_fill
            close_counts_tmp = deepcopy(close_counts)
            # Fill the other classes in a way that benefits class i most
            while num_fill < num_to_fill:
                assert np.all(close_counts_tmp >= 0)
                # Get classes that can still be filled except for i
                ind = np.setdiff1d(np.where(close_counts_tmp > 0)[0], i)
                # Find class with the smallest count
                ind_to_fill = ind[new_counts[ind].argmin()]
                new_counts[ind_to_fill] += 1
                close_counts_tmp[ind_to_fill] -= 1
                num_fill += 1
            assert new_counts.sum() == k
            max_count = new_counts.max()
            if new_counts[i] == max_count:
                y_pred.append(i)

        # label_counts = sure_counts + \
        #     np.minimum(close_counts, k - len(sure_indices))
        # Take labels that have more counts than half of k
        # return np.where(label_counts >= np.ceil(k / 2))[0]

        # close_counts = np.minimum(close_counts, k - len(sure_indices))
        # pred = []
        # for i, count in enumerate(sure_counts):
        #     if count + close_counts[i] >= sure_counts.max():
        #         pred.append(i)
        return np.array(y_pred)

    def classify(self, cell):
        """Returns majority label of <cell>. Returns all labels that tie."""
        counts = np.bincount(self.labels[list(cell)])
        max_counts = counts.max()
        return np.where(counts == max_counts)[0]

    def dist_to_facet(self, cur_cell, idx, facet, query, label, A, AAT, b,
                      params, is_adv, ub=None, dist_to_cell=None):

        if dist_to_cell is None:
            dist_to_cell = params['compute_dist_to_cell']
        if ub is None:
            ub = params['upperbound']

        if not dist_to_cell:
            # Apply screening
            b_hat = A @ query - b
            if np.any(ub < np.maximum(b_hat, 0)):
                self.log.debug('Screened.')
                return None
            # Compute distance to facet
            proj = gca(query, A, AAT, b, params, idx_plane=idx, ub=ub)
        else:
            # Get facets of neighbor cell that we want to compute distance to
            neighbor = self._get_neighbor_from_facet(cur_cell, facet)
            _, nb_hplanes = self.get_neighbor_facet(neighbor, params)
            # TODO: handle box constraint
            nb_A = nb_hplanes[:, :-1]
            nb_b = nb_hplanes[:, -1]
            # Apply screening
            b_hat = nb_A @ query - nb_b
            if np.any(ub < np.maximum(b_hat, 0)):
                self.log.debug('Screened.')
                return None
            nb_AAT = nb_A @ nb_A.T
            # Compute distance to cell
            proj = gca(query, nb_A, nb_AAT, nb_b, params, idx_plane=None, ub=ub)
        # Skip if solver exits with None, either facet is further
        # than ub or this is an invalid facet
        if proj is None:
            return None

        true_dist = self.compute_potential(query, label, proj, params)
        # Set lb bit to 0 since we compute true distance
        return (true_dist, (cur_cell, facet, is_adv, 0))

    def exit_func(self, exit_code, query, ub_facet, start_main, visited_cells,
                  computed_cells, dist=None, proj=None):
        if exit_code == self.MISCLF:
            self.log.info('CODE 3: Query is already misclassified.')
            return query, 0, exit_code
        self.log.info('FINISHED: main loop time: %.2f' %
                      (time.time() - start_main))
        self.log.info('num cells visited: %d, num cells computed: %d' %
                      (len(visited_cells), len(computed_cells)))
        if exit_code == self.FAIL:
            self.log.info('CODE 0: No valid adv cell found.')
            self.log.info('returning the initial upperbound.')
            return query, ub_facet[0], exit_code
        if exit_code == self.SUCCESS:
            self.log.info(
                'CODE 1: Success. Optimal adv cell found! Dist: %.4f' % dist)
            return proj, dist, exit_code
        if exit_code == self.TIMELIM:
            self.log.info('CODE 2: Time limit. At least one adv cell found! ' +
                          'Dist: %.4f' % dist)
            return proj, dist, exit_code
        raise NotImplementedError('Unknown exit code!')

    def get_neighbor_facet(self, cell, params, parallel=None):
        """Return facets of <cell>."""

        # generate set of potential neighbors to consider
        if params['neighbor_method'] == 'all':
            masks = [list(range(self.num_points)), ] * len(cell)
        elif params['neighbor_method'] == 'm_nearest':
            _, masks = self.approx_knn.search(
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
                if point in self.nb_dict:
                    union_nb.update(self.nb_dict[point])
                else:
                    # self.log.info('computing 1st-order neighbor of point: ', point)
                    # start = time.time()
                    self._get_1st_order_neighbors(
                        point, params, mask=masks[i], parallel=parallel)
                    union_nb.update(self.nb_dict[point])
                    # self.log.info('neighbors saved. finish in %.2fs.' %
                    #       (time.time() - start))
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
        hplanes = np.zeros((len_nb * len(cell), self.dim + 1), dtype=params['dtype'])
        facets = []
        for i, point in enumerate(cell):
            self._get_hyperplane(point, union_nb, hplanes[i * len_nb:(i + 1) * len_nb])
            for other in union_nb:
                facets.append((point, other))

        return facets, hplanes

    # ======================================================================= #
    #                              Find certificate                           #
    # ======================================================================= #

    def get_cert(self, query, label, params, k=None):
        """
        Compute adv certificate of <query>.
        """
        if k is None:
            k = self.k

        # create skip list to store facets described by
        # (key: distance, value: (cell, facet, is_adv, is_lb))
        Q = MySkipList()
        # the closest adv facet found so far
        ub_facet = (params['upperbound'], ([], (-1, -1), 0, 0))

        # get current cell of query: cur_cell
        _, indices = self.knn.search(query.reshape(1, -1), k)
        cur_cell = tuple(sorted(indices[0]))
        # define cells that GeoCert has visited
        visited_cells = set({cur_cell})
        # define cells that we have computed distance to
        computed_cells = set({cur_cell})

        # clabel = self.classify(cur_cell)
        # if self._check_labels_adv(clabel, label):
        #     return self.exit_func(self.MISCLF, query, ub_facet, 0,
        #                           visited_cells, computed_cells)
        start_main = time.time()

        # num_cores = multiprocessing.cpu_count()
        # with Parallel(n_jobs=params['num_cores'], prefer="threads") as parallel:
        # with Parallel(n_jobs=params['num_cores'], require='sharedmem') as parallel:
        parallel = None

        # ==================== Starting the main loop ======================= #
        while True:
            # handles exits with time limit
            if time.time() - start_main > params['time_limit']:
                self.log.info('Hit the time limit. Verifying seen adv cell...')
                while True:
                    if len(Q) > 0:
                        (dist, (cell, facet, is_adv, _)) = Q.popitem()
                    else:
                        # if Q is empty before finding any adversarial cell,
                        # return the failure code
                        return self.exit_func(self.FAIL, query, ub_facet,
                                              start_main, visited_cells,
                                              computed_cells)
                    if not is_adv:
                        continue
                    # DEBUG: If we check adv for sure anyways, we don't need to re-verify here
                    is_verified, proj = self.verify_adv_cell(query, label, cell, facet, k, params)
                    if is_verified:
                        dist = np.linalg.norm(query - proj)
                        return self.exit_func(
                            self.TIMELIM, query, ub_facet, start_main,
                            visited_cells, computed_cells, dist=dist, proj=proj)
                    # return self.exit_func(
                    #     self.TIMELIM, query, ub_facet, start_main,
                    #     visited_cells, computed_cells, dist=dist, proj=proj)

            # Determine neighboring facets of cur_cell
            self.log.debug('Getting facets...')
            facets, hplanes = self.get_neighbor_facet(
                cur_cell, params,
                parallel=parallel if params['parallel'] else None)

            # extract A, AA^T, and b from hplanes
            if params['compute_dist_to_cell']:
                A = None
                b = None
                AAT = None
            else:
                self.log.debug('Computing AAT...')
                A = hplanes[:, :-1]
                b = hplanes[:, -1]
                AAT = A @ A.T

            # ================= Start lower bound screening ================= #
            # Compute lb distance to facets, compute true distance if lb is
            # smaller than distance to lb_facet
            self.log.debug('Computing lb_dist...')
            lb_dist, proj = self._compute_lb_dist(query, hplanes, return_proj=True)
            self.log.debug('done.')

            self.log.debug('Screening with lower bound distance to facets...')
            # Sort facets in descending order of lower bound distance
            # NOTE: Flip sign of lb_dist
            lb_dist = - lb_dist
            indices = lb_dist.argsort()
            # Ignore facet further than ub_facet
            mask = lb_dist[indices] <= ub_facet[0] + self.TOL
            indices = indices[mask]

            # Create list of adv and benign facets that we will check
            self.log.debug('Determining adv/benign facets...')
            adv_indices, benign_indices = [], []
            adv_nb, benign_nb = [], []
            for idx in indices:
                # Find the neighboring cell
                neighbor = self._get_neighbor_from_facet(cur_cell, facets[idx])
                # Skip faet if neighbor is already visited
                if neighbor in visited_cells:
                    continue
                if ((params['treat_facet_as_cell'] or params['compute_dist_to_cell']) and neighbor in computed_cells):
                    continue
                # Check label of neighbor if adversarial
                neighbor_label = self.classify(neighbor)
                if self._check_labels_adv(neighbor_label, label):
                    adv_indices.append(idx)
                    adv_nb.append(neighbor)
                else:
                    benign_indices.append(idx)
                    benign_nb.append(neighbor)

            self.log.debug(
                f'After lb screening | num adv facets: {len(adv_indices)}, '
                f'num benign facets: {len(benign_indices)}')

            # Compute distance to adv facet first to set good upper bound
            self.log.debug('Computing distance to adv facets...')
            # Find k nearest neighbors to each adv projection
            _, proj_nn = self.knn.search(proj[adv_indices], k)
            self.log.debug('k-NN search done.')
            for i, (idx, neighbor) in enumerate(zip(adv_indices, adv_nb)):
                # Neighbor check here is not precise but is ok because it will
                # be checked again below
                if (not params['compute_dist_to_cell'] and set(proj_nn[i]) == set(neighbor)):
                    self.log.debug('Projection to bisector is nearest point.')
                    facet = (abs(lb_dist[idx]), (cur_cell, facets[idx], 1, 0))
                else:
                    # Find actual distance to the facet
                    if params['neighbor_method'] == 'all':
                        facet = self.dist_to_facet(
                            cur_cell, idx, facets[idx], query, label, A, AAT,
                            b, params, 1)
                    elif params['neighbor_method'] == 'm_nearest':
                        # Have to verify adv facet in approximate case because
                        # upper bound distance will be used for screening
                        is_verified, x_adv = self.verify_adv_cell(
                            query, label, cur_cell, facets[idx], k, params)
                        if is_verified:
                            dist = np.linalg.norm(query - x_adv)
                            facet = (dist, (cur_cell, facets[idx], 1, 0))
                        else:
                            facet = None

                self.log.debug((neighbor, facet))
                # Adv cell can be added to `computed_cells` as the distance is
                # computed exactly
                computed_cells.add(tuple(neighbor))
                # Skip if facet is invalid or further than the upper bound
                if (facet is None) or (facet[0] > ub_facet[0] + self.TOL):
                    continue
                Q.insert(facet[0], facet[1])
                # Set new upper bound of adversarial distance
                ub_facet = facet
                params['upperbound'] = facet[0]
                self.log.debug(f'>>>>>>> new ub_facet: {facet[0]:.4f} <<<<<<<')

            # Filter benign facets closer than ub_facet again since ub_facet
            # may be updated above
            self.log.debug('Second screening on benign facets...')
            mask = np.where(lb_dist[benign_indices] <= ub_facet[0] + self.TOL)[0]
            benign_indices = np.array(benign_indices)[mask]
            benign_nb = np.array(benign_nb)[mask]
            # ================== End lower bound screening ================== #

            if len(benign_indices) > 0:
                self.log.debug(f'Computing distance to {len(benign_indices)} benign facets...')
                # TODO: We can use parallel computation on benign facets because
                # we are no longer updating up_facet
                if params['parallel']:
                    facets_list = parallel(delayed(self.dist_to_facet)(
                        cur_cell, idx, facets[idx], query, label, A, AAT, b,
                        params, 0) for idx, neighbor in
                        zip(benign_indices, benign_nb))
                    for i, facet in enumerate(facets_list):
                        # Skip if facet is invalid or further than the upper bound
                        if (facet is None) or (facet[0] > ub_facet[0] + self.TOL):
                            continue
                        Q.insert(facet[0], facet[1])
                        computed_cells.add(tuple(benign_nb[i]))
                else:
                    _, proj_nn = self.knn.search(proj[benign_indices], k)
                    self.log.debug('k-NN search done.')
                    for i, (idx, neighbor) in enumerate(zip(benign_indices, benign_nb)):
                        if (not params['compute_dist_to_cell'] and set(proj_nn[i]) == set(neighbor)):
                            self.log.debug('Projection to bisector is nearest point.')
                            facet = (abs(lb_dist[idx]),
                                     (cur_cell, facets[idx], 0, 0))
                        else:
                            facet = self.dist_to_facet(
                                cur_cell, idx, facets[idx], query, label, A,
                                AAT, b, params, 0)
                        self.log.debug((neighbor, facet))
                        computed_cells.add(tuple(neighbor))
                        # Skip if facet is invalid or further than the upper bound
                        if (facet is None) or (facet[0] > ub_facet[0] + self.TOL):
                            continue
                        Q.insert(facet[0], facet[1])

            # Keep popping smallest item in Q until an unvisited cell is found
            while True:
                if len(Q) == 0:
                    self.log.info('PQ is empty. No facet to pop.')
                    return self.exit_func(
                        self.FAIL, query, ub_facet, start_main, visited_cells,
                        computed_cells)
                dist, (cell, facet, is_adv, _) = Q.popitem()
                neighbor = self._get_neighbor_from_facet(cell, facet)
                if neighbor not in visited_cells:
                    visited_cells.add(neighbor)
                    # Facet/cell is ignored if it's further than ub
                    if dist <= ub_facet[0] + self.TOL:
                        break

            if is_adv:
                # Verify the popped adv cell that it is valid and is adv
                is_verified, proj = self.verify_adv_cell(
                    query, label, cell, facet, k, params)
                if is_verified:
                    # If it is adv and verified then we are done
                    dist = np.linalg.norm(query - proj)
                    return self.exit_func(
                        self.SUCCESS, query, ub_facet, start_main,
                        visited_cells, computed_cells, dist=dist, proj=proj)
                self.log.info('adv cell is found but fails the verification.')
                self.log.info('continuing the search...')

            # Otherwise, look at the neighbor of the popped cell
            cur_cell = self._get_neighbor_from_facet(cell, facet)

    # ======================================================================= #
    #                            End find certificate                         #
    # ======================================================================= #

    def get_knn(self, query, k=None, return_distance=False):
        """Get k-nearest neighbors of query."""
        if query.ndim == 1:
            # queries = [query]
            queries = query.reshape(1, -1)
        else:
            queries = query
        knn = self.knn.search(queries, k)
        if return_distance:
            return knn[1], [np.array(dist) ** 2 for dist in knn[0]]
        return knn[1]

    def verify_adv_cell(self, query, label, cell, facet, k, params):
        """Verify that the cell neighboring `cell` via `facet` is adversarial, 
        i.e. has a different class from `label`.

        Args:
            query (np.array): Query/test point
            label (int): True label of `query`
            cell (tuple): Neighboring cell of the one to verify
            facet (tuple): Facet that neighbors the cell to verify
            k (int): k in k-NN
            params (dict): Main parameters

        Returns:
            bool: Whether the verified cell is adversarial
            np.array: The corresponding adversarial example
        """
        # Re-run the optimization to verify
        adv_cell = self._get_neighbor_from_facet(cell, facet)
        ver_params = deepcopy(params)
        ver_params['neighbor_method'] = 'all'
        ver_params['save_1nn_nb'] = False
        # DEBUG: ver_params?
        _, hplanes = self.get_neighbor_facet(adv_cell, params, parallel=None)
        A = hplanes[:, :-1]
        b = hplanes[:, -1]
        AAT = A @ A.T
        ver_params['max_proj_iters'] = params['max_proj_iters_verify']
        proj = gca(query, A, AAT, b, ver_params, idx_plane=None, ub=np.inf)
        if proj is None:
            return False, None
        # Get label more precisely
        final_label = self._get_precise_label(proj, k)
        is_adv = self._check_labels_adv(final_label, label)
        return is_adv, proj

    @classmethod
    def _compute_lb_dist(cls, query, hyperplanes, return_proj=False):
        """Compute naive lb distance from <query> to all <hyperplanes>.
        <signed_dist> is positive outside of cell."""
        signed_dist = hyperplanes[:, :-1] @ query - hyperplanes[:, -1]
        dist = signed_dist
        # dist = abs(signed_dist)
        if return_proj:
            # Move proj slightly inside the polytope
            proj = query - (signed_dist + 1e-5).reshape(
                -1, 1) * hyperplanes[:, :-1]
            return dist, proj
        return dist

    @staticmethod
    def _check_labels_adv(label_list, label):
        return np.any(label_list != label)

    @staticmethod
    def _get_neighbor_from_facet(cur_cell, facet):
        """
        Find a neighbor of <cur_cell> that shares same <facet>. 
        Assume one of point in <facet> is in <cur_cell>.
        """
        neighbor = [facet[1] if x == facet[0] else x for x in cur_cell]
        return tuple(sorted(neighbor))
