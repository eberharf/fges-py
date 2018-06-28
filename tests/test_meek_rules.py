import unittest
import networkx as nx

from meekrules import *
from graph_util import *

def make_graph(num_vertices, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(num_vertices)))
    graph.add_edges_from(edges)
    return graph

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



