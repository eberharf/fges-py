import unittest
import networkx as nx

from meekrules import MeekRules
from graph_util import *
from knowledge import Knowledge

def make_graph(num_vertices, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(num_vertices)))
    graph.add_edges_from(edges)
    return graph

def group_edges(edges):
    undirs = []
    dirs = []
    for e in edges:
        if e[::-1] in dirs:
            dirs.remove(e[::-1])
            undirs.append(e)
        else:
            dirs.append(e)
    return undirs, dirs

class SingleRuleTests(unittest.TestCase):

    def test_rule_1(self):
        # 0 --> 1 --- 2  =>  0 --> 1 --> 2
        g = make_graph(3, [(0, 1), (1, 2), (2, 1)])
        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        assert len(g.edges()) == 2
        assert has_dir_edge(g, 0, 1)
        assert has_dir_edge(g, 1, 2)

        # 0 --- 1 --> 2  =>  nothing
        g = make_graph(3, [(0, 1), (1, 0), (1, 2)])
        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        assert len(g.edges()) == 3
        assert has_undir_edge(g, 0, 1)
        assert has_dir_edge(g, 1, 2)

        # 0 --> 1 --- 2 and 0 -- 2  =>  nothing
        g = make_graph(3, [(0, 1), (1, 2), (2, 1), (0, 2), (2, 0)])
        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        assert len(g.edges()) == 5
        assert has_dir_edge(g, 0, 1)
        assert has_undir_edge(g, 0, 2)
        assert has_undir_edge(g, 1, 2)

    def test_rule_2(self):
        # 0 --> 1 --> 2 and 0 --- 2  => 0 --> 2
        g = make_graph(3, [(0, 1), (1, 2), (0, 2), (2, 0)])
        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        assert len(g.edges()) == 3
        assert has_dir_edge(g, 0, 1)
        assert has_dir_edge(g, 0, 2)
        assert has_dir_edge(g, 1, 2)

    def test_rule_3(self):
        '''
        0 --- 1, 0 --- 2, 0 --- 3, 1 --> 2 <-- 3  => 0 --> 2
        '''
        g = make_graph(4, [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
                           (1, 2), (3, 2)])

        assert is_kite(g, 2, 0, 1, 3)
        assert is_unshielded_non_collider(g, 3, 0, 1)

        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        assert len(g.edges()) == 7
        assert has_undir_edge(g, 0, 1)
        assert has_undir_edge(g, 0, 3)
        assert has_dir_edge(g, 0, 2)
        assert has_dir_edge(g, 1, 2)
        assert has_dir_edge(g, 3, 2)

        g = make_graph(4, [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0),
                           (1, 2), (3, 2), (1, 3)])

        assert is_kite(g, 2, 0, 1, 3)
        assert not is_unshielded_non_collider(g, 3, 0, 1)

        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        assert len(g.edges()) == 9
        assert has_undir_edge(g, 0, 1)
        assert has_undir_edge(g, 0, 2)
        assert has_undir_edge(g, 0, 3)
        assert has_dir_edge(g, 1, 2)
        assert has_dir_edge(g, 3, 2)
        assert has_dir_edge(g, 1, 3)

    def test_rule_4(self):
        g = make_graph(4, [(0, 1), (1, 0),
                           (0, 2), (2, 0),
                           (0, 3), (3, 0),
                           (1, 2),
                           (2, 3)])
        k = Knowledge()
        k.set_forbidden(1, 3)

        meek = MeekRules(undirect_unforced_edges=False, knowledge=k)
        meek.orient_implied_subset(g, [0])

        assert len(g.edges()) == 7
        assert has_dir_edge(g, 1, 2)
        assert has_dir_edge(g, 2, 3)
        assert has_dir_edge(g, 0, 3)
        assert has_undir_edge(g, 0, 1)
        assert has_undir_edge(g, 0, 2)


class LargeGraphTests(unittest.TestCase):

    def test_meek_graph_1a(self):
        g = make_graph(10, [
            (0, 3), (3, 0),
            (3, 6),
            (4, 6),
            (1, 4), (4, 1),
            (1, 2),
            (6, 7), (7, 6),
            (5, 7), (7, 5),
            (7, 8), (8, 7),
            (5, 8), (8, 5),
            (5, 2),
            (2, 9),
            (8, 9),
            (5, 9), (9, 5)
        ])

        meek = MeekRules(undirect_unforced_edges=False)
        meek.orient_implied(g)

        expected_undir_edges = [
            (0, 3),
            (1, 4),
            (5, 8),
        ]

        expected_dir_edges = [
            (3, 6),
            (4, 6),
            (1, 2),
            (6, 7),
            (7, 5),
            (7, 8),
            (5, 2),
            (2, 9),
            (8, 9),
            (5, 9),
        ]

        for e in expected_undir_edges:
            assert has_undir_edge(g, *e)
        for e in expected_dir_edges:
            assert has_dir_edge(g, *e)
        assert len(g.edges()) == len(expected_dir_edges) + 2 * len(expected_undir_edges)

    def test_meek_graph_1b(self):
        '''
        Add extra oriented edges, that should be unoriented
        '''
        g = make_graph(10, [
            (3, 0), #
            (3, 6),
            (4, 6),
            (1, 4), #
            (1, 2),
            (6, 7), (7, 6),
            (5, 7), (7, 5),
            (7, 8), (8, 7),
            (5, 8), #
            (5, 2),
            (2, 9),
            (8, 9),
            (5, 9), (9, 5)
        ])

        meek = MeekRules(undirect_unforced_edges=True)
        meek.orient_implied(g)

        expected_undir_edges = [
            (0, 3),
            (1, 4),
            (5, 8),
        ]

        expected_dir_edges = [
            (3, 6),
            (4, 6),
            (1, 2),
            (6, 7),
            (7, 5),
            (7, 8),
            (5, 2),
            (2, 9),
            (8, 9),
            (5, 9),
        ]

        for e in expected_undir_edges:
            assert has_undir_edge(g, *e)
        for e in expected_dir_edges:
            assert has_dir_edge(g, *e)
        assert len(g.edges()) == len(expected_dir_edges) + 2 * len(expected_undir_edges)

    def test_undirect_edges_1(self):
        g = make_graph(10, [
            (3, 0),  #
            (3, 6),
            (4, 6),
            (1, 4),  #
            (1, 2),
            (6, 7), (7, 6),
            (5, 7), (7, 5),
            (7, 8), (8, 7),
            (5, 8),  #
            (5, 2),
            (2, 9),
            (8, 9),
            (5, 9), (9, 5)
        ])

        meek = MeekRules()

        meek.undirect_unforced_edges_func(0, g)
        assert has_undir_edge(g, 0, 3)
        assert has_dir_edge(g, 3, 6)

        meek.undirect_unforced_edges_func(4, g)
        assert has_undir_edge(g, 4, 1)
        assert has_dir_edge(g, 4, 6)

        meek.undirect_unforced_edges_func(8, g)
        assert has_dir_edge(g, 8, 9)
        assert has_undir_edge(g, 8, 7)
        assert has_undir_edge(g, 8, 5)

if __name__ == "__main__":
    unittest.main()