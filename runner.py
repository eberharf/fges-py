import numpy as np
from SEMScore import *
from fges import *
import time
import sys

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

def main():
    dataset = load_file(sys.argv[1])
    # dataset = joblib.load("/home/ubuntu/PycharmProjects/fges-py/data/zebrafish_whole_brain2.npy")[:,:int(sys.argv[1])]
    score = SEMBicScore(dataset, len(dataset), float(sys.argv[2]))  # Initialize SEMBic Object
    sample_size = len(dataset)
    penalty = float(sys.argv[2])
    cov = np.cov(dataset, rowvar=False)

    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, 10, sys.argv[3], dataset, sample_size, penalty, cov)
    start_time = time.time()
    fges.search()
    print("--- %s seconds ---" % (time.time() - start_time))
    print(fges.graph.edges)

if __name__ == "__main__":
    main()
