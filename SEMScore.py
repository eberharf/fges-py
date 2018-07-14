import numpy as np
from numpy.linalg import inv
import math


class SEMBicScore:

    def __init__(self, dataset, penalty_discount):
        """ Initialize the SEMBicScore object. Assume that each
        row is a sample point, and each column is a variable
        """
        dataset = np.array(dataset)
        assert len(dataset.shape) == 2

        self.dataset = dataset
        self.cov = np.cov(dataset, rowvar=False)
        self.sample_size = len(dataset)
        self.penalty = penalty_discount
        self.corrcoef = np.corrcoef(dataset.transpose())
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def partial_corr(self, x, y, Z):
        """
        Returns the partial correlation coefficients between elements of X controlling for the elements in Z.
        """
        x_data = self.dataset[:,x]
        y_data = self.dataset[:,y]
        if Z == []:
            return self.corrcoef[x, y]
        Z_data = self.dataset[:,Z]

        beta_i = np.linalg.lstsq(Z_data, x_data)[0]
        beta_j = np.linalg.lstsq(Z_data, y_data)[0]

        res_j = x_data - Z_data.dot(beta_i)
        res_i = y_data - Z_data.dot(beta_j)

        return np.corrcoef(res_i, res_j)[0, 1]

    def recursive_partial_corr(self, x, y, Z):
        if len(Z) == 0:
            return self.corrcoef[x, y]
        else:
            z0 = min(Z)
            Z1 = Z - {z0}
            key1 = (frozenset({x, y}), Z1)
            if key1 not in self.cache:
                term1 = self.recursive_partial_corr(x, y, Z1)
                if len(Z1) > 0:
                    self.cache[key1] = term1
            else:
                self.cache_hits += 1
                term1 = self.cache[key1]

            key2 = (frozenset({x, z0}), Z1)
            if key2 not in self.cache:
                term2 = self.recursive_partial_corr(x, z0, Z1)
                if len(Z1) > 0:
                    self.cache[key2] = term2
            else:
                self.cache_hits += 1
                term2 = self.cache[key2]

            key3 = (frozenset({y, z0}), Z1)
            if key3 not in self.cache:
                term3 = self.recursive_partial_corr(y, z0, Z1)
                if len(Z1) > 0:
                    self.cache[key3] = term3
            else:
                self.cache_hits += 1
                term3 = self.cache[key3]

            return (term1 - (term2 * term3)) / math.sqrt((1 - (term2 * term2)) * (1 - (term3 * term3)))

    def local_score(self, node, parents):
        """ `node` is an int index """
        #print("Node:", node, "Parents:", parents)
        variance = self.cov[node][node]
        p = len(parents)

        # np.ix_(rowIndices, colIndices)

        covxx = self.cov[np.ix_(parents, parents)]
        covxx_inv = inv(covxx)

        if (p == 0):
            covxy = []
        else:
            # vector
            covxy = self.cov[np.ix_(parents, [node])]

        b = np.dot(covxx_inv, covxy)

        variance -= np.dot(np.transpose(covxy), b)

        if variance <= 0:
            return None

        returnval = self.score(variance, p);
        return returnval

    def local_score_no_parents(self, node):
        """ if node has no parents """
        variance = self.cov[node][node]

        if variance <= 0:
            return None

        return self.score(variance)

    def score(self, variance, parents_len=0):
        #print("Variance:", variance)
        #print(parents_len)
        bic = - self.sample_size * math.log(variance) - parents_len * self.penalty * math.log(self.sample_size)
        # TODO: Struct prior?
        #print(bic)

        return bic

    def local_score_diff_parents(self, node1, node2, parents):
        #print(node1, node2, parents)
        parents = frozenset(parents)
        r = self.recursive_partial_corr(node1, node2, parents)
        #print(r)
        return -self.sample_size * math.log(1.0 - r * r) - (len(parents) + 2) * self.penalty * math.log(self.sample_size)
        #return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        r = self.corrcoef[node1][node2]
        #print(r)
        return -self.sample_size * math.log(1.0 - r * r)
        #return self.local_score(node2, [node1]) - self.local_score(node2, [])
