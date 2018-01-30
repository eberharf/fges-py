import numpy as np
from SEMScore import *
from fges import *
import time
import sys

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

def main():
    dataset = load_file(sys.argv[1])
    score = SEMBicScore(dataset, len(dataset), float(sys.argv[2])) # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, 10, sys.argv[3]) 
    start_time = time.time()
    fges.search()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
