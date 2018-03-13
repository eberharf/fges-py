
from fges import FGES
from SEMScore import SEMBicScore
from search_util import *
import numpy as np

class SemEstimator:

    def __init__(self, dataset, sparsity=2, savename=None):
        ''' Initialize a SEM Estimator '''
        self.dataset = dataset
        self._score = SEMBicScore(dataset, sparsity)
        self._fges = FGES(list(range(dataset.shape[1])), self._score, 10, savename)

        self.pattern = None
        self.dag = None
        self.params = None
        self.residuals = None
        self.graph_cov = None
        self.true_cov = np.cov(dataset.transpose())

    def search(self):
        ''' Run an FGES search '''
        self._fges.search()
        self.pattern = self._fges.graph
        self.dag = dagFromPattern(self._fges.graph)

    def estimate(self):
        ''' Estimate edge weights '''
        if self.dag is None:
            self.search()

        self.params, self.residuals = estimate_parameters(self.dag, self.dataset)
        self.graph_cov = get_covariance_matrix(self.params, self.residuals)

