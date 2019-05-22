from graph_util import *
import networkx as nx
import unittest

class TestAncestors(unittest.TestCase):

    def test_simple_graph(self):
        '''
        Test a simple graph X0 -> X1 -> X2 -> X3
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        for i in [0, 1, 2]:
            graph.add_edge(i, i + 1)

        for i in range(4):
            assert get_ancestors(graph, i) == set(range(i + 1))

    def test_cyclic_graph(self):
        '''
        X0 -> {X1, X2} -> X3
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3])
        graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])

        assert get_ancestors(graph, 0) == set([0])
        assert get_ancestors(graph, 1) == set([0, 1])
        assert get_ancestors(graph, 2) == set([0, 2])
        assert get_ancestors(graph, 3) == set([0, 1, 2, 3])

    def test_undirected_edges(self):
        '''
        Ancestors should only consider directed edges
        X0 -> X1 -> X2 -- X3 --> X4
        '''
        graph = nx.DiGraph()
        graph.add_nodes_from([0, 1, 2, 3, 4])
        graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 2), (3, 4)])

        assert get_ancestors(graph, 0) == set([0])
        assert get_ancestors(graph, 1) == set([0, 1])
        assert get_ancestors(graph, 2) == set([0, 1, 2])
        assert get_ancestors(graph, 3) == set([3])
        assert get_ancestors(graph, 4) == set([3, 4])

if __name__ == "__main__":
    unittest.main()