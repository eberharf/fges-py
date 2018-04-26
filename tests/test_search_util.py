from search_util import *
from graph_util import *
import networkx as nx
import unittest
from SemEstimator import SemEstimator

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

    def test_cycle(self):
        '''
        X0 -- X1 -- X2 -- X3 -- X0
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 0),
                              (1, 2), (2, 1),
                              (2, 3), (3, 2),
                              (3, 0), (0, 3)])
        dag = dagFromPattern(graph)
        assert dag is None

class TestDagFromPatternWithColliders(unittest.TestCase):

    def test_triple(self):
        '''
        X0 -- X1 -- X2
        Resulting graph shouldn't have X1 be a collider
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edges_from([(0, 1), (1, 0), (1, 2), (2, 1)])

        dag, penalty = dagFromPatternWithColliders(graph)

        assert dag is not None
        assert penalty == 0
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

        dag, penalty = dagFromPatternWithColliders(graph)

        assert dag is not None
        assert penalty == 0

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

        dag, penalty = dagFromPatternWithColliders(graph)

        assert dag is not None
        assert penalty == 0

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

        dag, penalty = dagFromPatternWithColliders(graph)

        assert dag is not None
        assert penalty == 0

        assert get_undir_edge(dag) is None
        assert not is_def_collider(dag, 0, 1, 2)
        assert is_def_collider(dag, 1, 2, 3)
        assert not is_def_collider(dag, 2, 3, 4)

    def test_cycle(self):
        '''
        X0 -- X1 -- X2 -- X3 -- X0
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 0),
                              (1, 2), (2, 1),
                              (2, 3), (3, 2),
                              (3, 0), (0, 3)])
        dag, penalty = dagFromPatternWithColliders(graph)

        assert dag is not None
        assert penalty == 1
        assert not detect_cycle(dag)

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

        estimator = SemEstimator(dataset)
        estimator.estimate()

        print(dataset.shape)

        true_params = np.array([[0, 1, 2, 0],
                                [0, 0, 0, 0.5],
                                [0, 0, 0, -1],
                                [0, 0, 0, 0]])

        true_pred = np.matmul(dataset, true_params)
        true_errors = true_pred - dataset
        true_variances = np.var(true_errors, axis=0)

        print("Estimated Parameters:\n", estimator.params)
        print("True Parameters:\n", true_params)

        print("Graph Error Variances:\n", estimator.residuals.diagonal())
        print("True Error Variances:\n", true_variances)

        print("Graph Covariance:\n", estimator.graph_cov)
        print("True Covariance:\n", estimator.true_cov)

    def test_cycle(self):
        '''Test cycle X0 --- X1 --- X2 --- X3 --- X0'''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (1, 0),
                              (1, 2), (2, 1),
                              (2, 3), (3, 2),
                              (3, 0), (0, 3)])
        assert dagFromPattern(graph) is None

        graph.remove_edge(0, 1)
        graph.remove_edge(0, 3)
        new_dag = dagFromPattern(graph)
        assert new_dag is not None
        assert set(new_dag.edges()) == set([(1, 0), (1, 2), (2, 3), (3, 0)])

if __name__ == "__main__":
    unittest.main()