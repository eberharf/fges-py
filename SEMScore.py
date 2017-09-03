import numpy as np
from numpy.linalg import inv
import math 

class SEMBicScore:

	def __init__(self, dataset, sampleSize, penaltyDiscount):
		""" Initialize the SEMBicScore. Assume that each 
		row is a sample point, and each column is a variable
		"""
		self.cov = np.cov(dataset, rowVar = False)
		self.sampleSize = sampleSize
		self.penalty = penaltyDiscount

	def localScore(node, parents):
		# TODO: Handle singular matrix 
		""" `node` is an int index """
		variance = cov[node][node]
		numParents = len(parents)

		# np.ix_(rowIndices, colIndices)
		covxx = self.cov[np.ix_(parents,parents)]
		covxxInv = inv(covxx)

		covxy = self.cov[np.ix_(parents, node)]

		b = np.dot(covxxInv, covxy)

		variance -= np.dot(covxy, b)

		if variance <= 0:
			return None 

		return self.score(variance)

	def localScore(node):
		""" if node has no parents """
		variance = cov[node][node]

		if variance <= 0:
			return None

		return self.score(variance)

	def score(variance):
		bic = - self.sampleSize * math.log(variance) - self.penalty * (self.penalty + 1) * math.log(self.sampleSize)
		# TODO: Struct prior?
		return bic

	def localScoreDiff(node1, node2, parents):
		return localScore(node2, parents + node1) - localScore(node2, parents)

	def localScoreDiff(node1, node2):
		return localScore(node2, node1) - localScore(node2)

