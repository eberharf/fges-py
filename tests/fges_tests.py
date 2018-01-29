from SEMScore import *
from fges import *
import unittest

def run_fges(data_file):
    '''
    Run FGES on a data file, returning the resulting graph
    :param data_file: the data file to use
    :return: DiGraph from fges.search()
    '''
    dataset = np.loadtxt(data_file, skiprows = 1)
    score = SEMBicScore(dataset, len(dataset), 2)  # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, 10)
    return fges.search()

def assert_unoriented_edge(edges, e):
    # TODO: should this be an `or` and an `and`?
    assert (e in edges) or (e[::-1] in edges)

def assert_oriented_edge(edges, e):
    assert (e in edges) and not (e[::-1] in edges)

class SimpleTests(unittest.TestCase):

    def test_collider_1(self):
        '''
        Test a simple collider X1 -> X3 <- X2
        FGES should obtain the exact graph
        '''
        graph = run_fges("../data/collider_1.txt")
        edges = graph.edges

        assert_oriented_edge(edges, (0, 2))
        assert_oriented_edge(edges, (1, 2))

    def test_collider_2(self):
        '''
        Test a collider with common cause
         X1 -> X2 -> X3
         X1 -> X4 -> X3
        FGES should resolve collider at X3, and connection between other nodes
        '''
        graph = run_fges("../data/collider_2.txt")
        edges = graph.edges

        assert_oriented_edge(edges, (1, 2))
        assert_oriented_edge(edges, (3, 2))
        assert_unoriented_edge(edges, (0, 1))
        assert_unoriented_edge(edges, (0, 3))

    def test_collider_3(self):
        '''
        Graph Nodes: X2,X3,X4,X1,X5,X6
        Graph Edges:
          {X2, X4} --> X3 --> {X1, X5} --> X6
            1. X1 --> X6
            2. X2 --> X3
            3. X3 --> X1
            4. X3 --> X5
            5. X4 --> X3
            6. X5 --> X6
        FGES should orient all edges
        '''
        graph = run_fges("../data/collider_3.txt")
        edges = graph.edges

        #print(edges)

        # Extra edges: (1, 5), (3, 4), (4, 3), (5, 3), (5, 4)]
        #  X3 --> X6 (skip shielding)
        #  X1 --- X5 (extra hidden common cause)
        #  X6 --- {X1, X5} (fails to orient collider)

        for e in [(3, 5), (0, 1), (1, 3), (1, 4), (2, 1), (4, 5)]:
            assert_oriented_edge(edges, e)


if __name__ == "__main__":
    unittest.main()