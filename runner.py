import numpy as np
from SEMScore import *
from fges import *

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

dataset = load_file("data/data.1.txt")
score = SEMBicScore(dataset, len(dataset), 0.05)
variables = list(range(len(dataset[0])))
fges = FGES(variables, score, 10)
fges.search()