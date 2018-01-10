import numpy as np
from SEMScore import *
from fges import *
import time

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

dataset = load_file("data/HCP.tsv")
score = SEMBicScore(dataset, len(dataset), 10) # Penalty discount is third argument here
variables = list(range(len(dataset[0])))
fges = FGES(variables, score, 10)
start_time = time.time()
fges.search()
print("--- %s seconds ---" % (time.time() - start_time))