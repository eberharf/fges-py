import numpy as np
from numpy.linalg import inv
import math
import resource

MEMORY_LIMIT = 8 * 1024**2 # KB

class SEMBicScore:

    def __init__(self,
                 penalty_discount,
                 dataset=None,
                 corrs=None, dataset_size=None,
                 cache_interval=1,
                 prior=None,
                 prior_weight=None,
                 prior_forward_only=False,
                 max_depth=8,
                 use_big_param_penalty=False):
        """Initialize the SEMBicScore object.

        Must specify either the dataset or the correlation matrix and dataset size.

        :param penalty_discount: Sparsity parameter
        :param dataset: 2D np.array, with a row for each data point and a column for each variable
        :param corrs: 2D square np.array, the correlation coefficients
        :param dataset_size: the number of data points
        :param cache_interval: parameter to limit how many partial correlations are cached
        """
        self.penalty = penalty_discount
        self.dataset = dataset
        self.cache = {}
        self.cache_interval = cache_interval
        self.cache_too_big = False
        self.prior = prior
        self.prior_weight = prior_weight
        self.prior_forward_only = prior_forward_only
        self.in_forward = True
        self.max_depth = max_depth

        if use_big_param_penalty:
            self.param_penalty_weight = lambda parents: len(parents) + 2
        else:
            self.param_penalty_weight = lambda _: 1

        if dataset is not None:
            assert corrs is None and dataset_size is None, \
                "SEMBicScore: must specify either dataset or {corrs, dataset_size}"
            assert len(dataset.shape) == 2
            self.corrcoef = np.corrcoef(dataset.transpose())
            self.sample_size = dataset.shape[0]

        elif corrs is not None and dataset_size is not None:
            assert len(corrs.shape) == 2
            assert corrs.shape[0] == corrs.shape[1]
            self.corrcoef = corrs
            self.sample_size = dataset_size

        else:
            raise AssertionError("SEMBicScore: must specify either dataset or {corrs, dataset_size}")

        if prior is not None:
            assert prior_weight is not None
            assert prior.shape == self.corrcoef.shape
            self.effective_prior = prior_weight * np.log(prior / (1 - prior))

    def set_in_forward(self, val):
        self.in_forward = val

    def partial_corr(self, x, y, Z):
        """
        Returns the partial correlation coefficients between elements of X controlling for the elements in Z.
        """
        Z = list(Z)
        x_data = self.dataset[:,x]
        y_data = self.dataset[:,y]
        if Z == []:
            return self.corrcoef[x, y]
        Z_data = self.dataset[:,Z]

        beta_i = np.linalg.lstsq(Z_data, x_data, rcond=None)[0]
        beta_j = np.linalg.lstsq(Z_data, y_data, rcond=None)[0]

        res_j = x_data - Z_data.dot(beta_i)
        res_i = y_data - Z_data.dot(beta_j)

        return np.corrcoef(res_i, res_j)[0, 1]

    def recursive_partial_corr(self, x, y, Z, depth=0):
        if len(Z) == 0:
            return self.corrcoef[x, y]

        k = (frozenset({x, y}), Z)
        if k in self.cache:
            return self.cache[k]

        if depth >= self.max_depth:
            return self.partial_corr(x, y, Z)

        if not self.cache_too_big:
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > MEMORY_LIMIT
            if r:
                print("Locking cache")
                self.cache_too_big = True

        
        z0 = min(Z)
        Z1 = Z - {z0}
        term1 = self.recursive_partial_corr(x, y, Z1, depth + 1)
        term2 = self.recursive_partial_corr(x, z0, Z1, depth + 1)
        term3 = self.recursive_partial_corr(y, z0, Z1, depth + 1)

        answer = (term1 - (term2 * term3)) / math.sqrt((1 - (term2 * term2)) * (1 - (term3 * term3)))

        if not self.cache_too_big and len(Z) % self.cache_interval == 0:
            self.cache[k] = answer

        return answer

    # def local_score(self, node, parents):
    #     """ `node` is an int index """
    #     #print("Node:", node, "Parents:", parents)
    #     variance = self.cov[node][node]
    #     p = len(parents)
    #
    #     # np.ix_(rowIndices, colIndices)
    #
    #     covxx = self.cov[np.ix_(parents, parents)]
    #     covxx_inv = inv(covxx)
    #
    #     if (p == 0):
    #         covxy = []
    #     else:
    #         # vector
    #         covxy = self.cov[np.ix_(parents, [node])]
    #
    #     b = np.dot(covxx_inv, covxy)
    #
    #     variance -= np.dot(np.transpose(covxy), b)
    #
    #     if variance <= 0:
    #         return None
    #
    #     returnval = self.score(variance, p)
    #     return returnval

    # def local_score_no_parents(self, node):
    #     """ if node has no parents """
    #     variance = self.cov[node][node]
    #
    #     if variance <= 0:
    #         return None
    #
    #     return self.score(variance)

    # def score(self, variance, parents_len=0):
    #     bic = - self.sample_size * math.log(variance) - parents_len * self.penalty * math.log(self.sample_size)
    #     # TODO: Struct prior?
    #     return bic

    def local_score_diff_parents(self, node1, node2, parents):
        # return self.local_score(node2, parents + [node1]) - self.local_score(node2, parents)

        parents = frozenset(parents)
        r = self.recursive_partial_corr(node1, node2, parents)
        answer = -self.sample_size * np.log(1.0 - r**2) - self.param_penalty_weight(parents) * self.penalty * np.log(self.sample_size)

        if self.prior is not None and \
                (not self.prior_forward_only or self.in_forward):
            answer += self.effective_prior[node1, node2]

        return answer

    def local_score_diff(self, node1, node2):
        return self.local_score_diff_parents(node1, node2, [])

        # r = self.corrcoef[node1][node2]
        # return -self.sample_size * math.log(1.0 - r * r)

        # return self.local_score(node2, [node1]) - self.local_score(node2, []) 
