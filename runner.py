import numpy as np
from SEMScore import *
from fges import *
import time
import sys

import joblib
import dill
import pickle
import argparse

parser = argparse.ArgumentParser(description="Run FGES.")
parser.add_argument("dataset", type=str,
                    help="File to load data from (text file with a " \
                         "row for each datapoint and a column for each variable. \
                         If name of file ends in -checkpoint, this is assumed to be  \
                         a saved checkpoint instead")
parser.add_argument("sparsity", type=float, help="Sparsity penalty to use.")
parser.add_argument("save_frequency", type=int, help="Frequency to checkpoint work. Set to 0 to turn off checkpointing.
parser.add_argument("save_name", type=str, help="File to save output to.")


def load_file(data_file):
    return np.loadtxt(data_file, skiprows = 1)


def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def main():
    args = parser.parse_args()
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

                    
def load_pickled_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def main():
    args = parser.parse_args()
    if sys.argv[1][-15:] == "-checkpoint.pkl":
        start_time = time.time()
        fges = load_checkpoint(args.dataset)
        fges.last_checkpoint = time.time()
        result = fges.search(checkpoint=True)
    else:
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
