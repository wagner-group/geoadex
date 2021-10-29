'''Implement table to store edge distance'''
import numpy as np


class NeighborTable:

    def __init__(self, num_points):
        self.tab = np.zeros(num_points, num_points, 2) - 1
        # store current nearest edge and its distance
        self.current_nearest = ([-1, -1], -1)

    def _check_edge(self, edge):
        if not isinstance(edge, tuple):
            raise TypeError('expect <edge> to be a tuple.')
        if len(edge) != 2:
            raise TypeError('expect <edge> to have 2 entries.')
        if edge[0] > edge[1]:
            return edge
        return (edge[1], edge[0])

    def get(self, edge):
        edge = self._check_edge(edge)
        return self.tab[edge[0], edge[1]]

    def set(self, edge, dist, lb):
        edge = self._check_edge(edge)
        self.tab[edge[0], edge[1], 0] = dist
        self.tab[edge[0], edge[1], 1] = lb
