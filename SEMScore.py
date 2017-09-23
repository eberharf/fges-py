import numpy as np
from numpy.linalg import inv
import math


class SEMBicScore:

    def __init__(self, dataset, sample_size, penalty_discount):
        """ Initialize the SEMBicScore. Assume that each 
        row is a sample point, and each column is a variable
        """
        self.cov = np.cov(dataset, rowVar=False)
        self.sample_size = sample_size
        self.penalty = penalty_discount

    def local_score(self, node, parents):
        # TODO: Handle singular matrix
        """ `node` is an int index """
        variance = self.cov[node][node]

        # np.ix_(rowIndices, colIndices)
        covxx = self.cov[np.ix_(parents, parents)]
        covxx_inv = inv(covxx)

        covxy = self.cov[np.ix_(parents, node)]

        b = np.dot(covxx_inv, covxy)

        variance -= np.dot(covxy, b)

        if variance <= 0:
            return None

        return self.score(variance)

    def local_score_no_parents(self, node):
        """ if node has no parents """
        variance = self.cov[node][node]

        if variance <= 0:
            return None

        return self.score(variance)

    def score(self, variance):
        bic = - self.sample_size * \
            math.log(variance) - self.penalty * \
            (self.penalty + 1) * math.log(self.sample_size)
        # TODO: Struct prior?
        return bic

    def local_score_diff_parents(self, node1, node2, parents):
        return self.local_score(node2, parents + node1) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        return self.local_score(node2, node1) - self.local_score_no_parents(node2)
