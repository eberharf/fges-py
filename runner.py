import numpy as np
from SEMScore import *
from fges import *
import time
import sys
import joblib
import dill
import pickle

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    if sys.argv[1][-15:] == "-checkpoint.pkl":
        start_time = time.time()
        fges = load_checkpoint(sys.argv[1])
        fges.last_checkpoint = time.time()
        result = fges.search(checkpoint=True)
    else:
        dataset = load_file(sys.argv[1])
        score = SEMBicScore(dataset, float(sys.argv[2])) # Initialize SEMBic Object
        variables = list(range(len(dataset[0])))
        print("Running FGES on graph with " + str(len(variables)) + " nodes.")
        fges = FGES(variables, score, sys.argv[1], sys.argv[2], save_frequency=int(sys.argv[3]), save_name=sys.argv[4])
        start_time = time.time()
        result = fges.search()
    print("--- %s seconds ---" % (time.time() - start_time))
    with open(sys.argv[4] + '.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
