import numpy as np
from SEMScore import *
from fges import *
import time
import sys
import argparse
import pickle

parser = argparse.ArgumentParser(description="Run FGES.")
parser.add_argument("dataset", type=str,
                    help="File to load data from (text file with a " \
                         "row for each datapoint and a column for each variable.")
parser.add_argument("sparsity", type=float, help="Sparsity penalty to use.")
parser.add_argument("save_name", type=str, help="File to save output to.")

def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)

def load_pickled_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def main():
    args = parser.parse_args()
    dataset = load_file(args.dataset)
    score = SEMBicScore(dataset, args.sparsity) # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, args.dataset, args.sparsity)
    start_time = time.time()
    result = fges.search()
    print("--- %s seconds ---" % (time.time() - start_time))
    with open(args.save_name + '.pkl', 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
