from search_util import *
from graph_util import *
import networkx as nx
import unittest
from fges import *
from SEMScore import *

class TestDagFromPattern(unittest.TestCase):

    def test_triple(self):
        '''
        X0 -- X1 -- X2
        Resulting graph shouldn't have X1 be a collider
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1)])

        dag = dagFromPattern(graph)

        assert get_undir_edge(dag) is None
        assert not is_def_collider(dag, 0, 1, 2)

    def test_cross(self):
        '''X0 -- X1 -- X2 and X3 -- X1 -- X4'''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edges_from([(0, 1), (1, 0),
                              (1, 2), (2, 1),
                              (3, 1), (1, 3),
                              (1, 4), (4, 1)])

        dag = dagFromPattern(graph)

        assert get_undir_edge(dag) is None
        for i in [0, 2, 3, 4]:
            for j in [0, 2, 3, 4]:
                if i != j:
                    assert not is_def_collider(dag, i, 1, j)

    def test_undo(self):
        '''
        X0 --- X1 <-- X2
        The algorithm will first try X0 --> X1, then have to
          retry since this creates a collider.
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 0), (2, 1)])

        dag = dagFromPattern(graph)

        assert get_undir_edge(dag) is None
        assert not is_def_collider(dag, 0, 1, 2)
        assert has_dir_edge(dag, 1, 0)

    def test_several(self):
        '''
        X0 --- X1 --> X2 <-- X3 --- X4
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edges_from([(0, 1), (1, 0),
                              (1, 2),
                              (3, 2),
                              (3, 4), (4, 3)])

        dag = dagFromPattern(graph)

        assert get_undir_edge(dag) is None
        assert not is_def_collider(dag, 0, 1, 2)
        assert is_def_collider(dag, 1, 2, 3)
        assert not is_def_collider(dag, 2, 3, 4)

class TestEstimateParameters(unittest.TestCase):

    def test_params_1(self):
        '''
        Graph Edges:
            1. X1 --> X2 w =  1.0
            2. X1 --> X3 w =  2.0
            3. X2 --> X4 w =  0.5
            4. X3 --> X4 w = -1.0
        '''
        dataset = np.loadtxt("../data/params_1.txt", skiprows=1)
        score = SEMBicScore(dataset, len(dataset), 2)
        variables = list(range(len(dataset[0])))
        fges = FGES(variables, score, 10, "test.npy")
        fges.search()

        dag = dagFromPattern(fges.graph)
        params, residuals = estimate_parameters(dag, dataset)

        print("True Parameters:\n",
              np.array([[1, 2, 0, 0],
                        [0, 0, 0, 0.5],
                        [0, 0, 0, -1],
                        [0, 0, 0, 0]]))

        print("Estimated Parameters:\n", params)

        print("Residuals:\n", residuals)

if __name__ == "__main__":
    unittest.main()