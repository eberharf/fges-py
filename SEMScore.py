import numpy as np
from numpy.linalg import inv
import math
from scipy.stats.stats import pearsonr


class SEMBicScore:

    def __init__(self, dataset, sample_size, penalty_discount):
        """ Initialize the SEMBicScore. Assume that each 
        row is a sample point, and each column is a variable
        """
        self.dataset = dataset
        self.cov = np.cov(dataset, rowvar=False)
        self.sample_size = sample_size
        self.penalty = penalty_discount

    def partial_corr(self, x, y, Z):
        """
        Returns the partial correlation coefficients between elements of X controlling for the elements in Z.
        """
        x = self.dataset[:,x]
        y = self.dataset[:,y]
        if Z == []:
            return pearsonr(x,y)
        Z = self.dataset[:,Z]

        beta_i = np.linalg.lstsq(Z, x)[0]
        beta_j = np.linalg.lstsq(Z, y)[0]

        res_j = x - Z.dot(beta_i)
        res_i = y - Z.dot(beta_j)

        corr = np.corrcoef(res_i, res_j)

        return corr.item(0, 0)

    def local_score(self, node, parents):
        # TODO: Handle singular matrix
        """ `node` is an int index """
        print("Node:", node, "Parents:", parents)
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

        variance -= np.dot(covxy, b)

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

    def score(self, variance, parents_len):
        #print("Variance:", variance)
        bic = - self.sample_size * math.log(variance) - self.penalty * math.log(self.sample_size)
        # TODO: Struct prior?
        print(bic)
        return bic

    def local_score_diff_parents(self, node1, node2, parents):
        print(node1, node2, parents)
        #return self.score(self.partial_corr(node1, node2, parents), len(parents))
        return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

    def local_score_diff(self, node1, node2):
        #print(self.partial_corr(node1, node2, []))
        #return self.score(self.partial_corr(node1, node2, []), 0)
        return self.local_score(node2, [node1]) - self.local_score(node2, [])
