import numpy as np
from pyskiplist import SkipList


class MySkipList(SkipList):

    def search_below(self, query):
        """
        Search the skip list and return all key-value pairs with key equal to
        or smaller than a given <query>.
        """

        # first check if <query> is in the list
        try:
            idx = self.index(query)
            return self[:idx + 1]
        except Exception as e:
            pass

        length = len(self)
        lo = 0
        hi = length - 1
        while hi - lo != 1:
            cur = int((lo + hi) / 2)
            cur_val = self[cur][0]
            if cur_val <= query:
                lo = cur
            else:
                hi = cur

        if self[lo][0] <= query <= self[hi][0]:
            return self[:hi]
        return self[:0]
