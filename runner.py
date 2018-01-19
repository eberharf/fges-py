import numpy as np
from SEMScore import *
from fges import *
import time

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

if __name__ == '__main__':
    dataset = load_file("data/HCP.tsv")[:,:10]
    score = SEMBicScore(dataset, len(dataset), 2) # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, 10)
    start_time = time.time()
    fges.search()
    print("--- %s seconds ---" % (time.time() - start_time))