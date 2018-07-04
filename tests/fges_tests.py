from SEMScore import *
from fges import *
from tests.test_util import *
import unittest

def run_fges_file(data_file, **kwargs):
    '''
    Run FGES on a data file, returning the resulting graph
    :param data_file: the data file to use
    :return: DiGraph from fges.search()
    '''
    return run_fges_array(np.loadtxt(data_file, skiprows = 1), **kwargs)

def run_fges_array(dataset, **kwargs):
    '''
    Run FGES on a loaded array, with a row for each datapoint and a column for each variable.
    :param dataset: numpy array
    :return: DiGraph from fges.search()
    '''
    score = SEMBicScore(dataset, 2)  # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, 10, **kwargs)
    return fges.search()

def assert_unoriented_edge(edges, e):
    assert (e in edges) and (e[::-1] in edges)

def assert_oriented_edge(edges, e):
    assert (e in edges) and not (e[::-1] in edges)

class SimpleTests(unittest.TestCase):

    def test_collider_1(self):
        '''
        Test a simple collider X1 -> X3 <- X2
        FGES should obtain the exact graph
        '''
        result = run_fges_file("../data/collider_1.txt")
        edges = result['graph'].edges()

        assert_oriented_edge(edges, (0, 2))
        assert_oriented_edge(edges, (1, 2))

    def test_collider_2(self):
        '''
        Test a collider with common cause
         X1 -> X2 -> X3
         X1 -> X4 -> X3
        FGES should resolve collider at X3, and connection between other nodes
        '''
        result = run_fges_file("../data/collider_2.txt")
        edges = result['graph'].edges()

        assert_oriented_edge(edges, (1, 2))
        assert_oriented_edge(edges, (3, 2))
        assert_unoriented_edge(edges, (0, 1))
        assert_unoriented_edge(edges, (0, 3))

    def test_collider_3(self):
        '''
        Graph Edges:
          {X0, X1} --> X2 --> {X3, X4} --> X5

        FGES should orient all edges
        '''
        result = run_fges_file("../data/collider_3.txt")
        edges = result['graph'].edges()

        print("Computed Edges:", edges)

        expected = [(0, 2), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)]
        all_correct = True

        for e in expected:
            if e not in edges:
                all_correct = False
                print("Missing edge", e, "({} -> {})".format(*e))

        for e in edges:
            if e not in expected:
                all_correct = False
                print("Extra edge", e, "({} -> {})".format(*e))

        assert all_correct

    def test_collider_4(self):
        '''
        Graph Edges:
          {X1, X2} --> X3 --> X4 --> X5

        FGES should orient all edges
        '''
        result = run_fges_file("../data/collider_4.txt")
        edges = result['graph'].edges()

        expected = [(0, 2), (1, 2), (2, 3), (3, 4)]

        for e in expected:
            assert_oriented_edge(edges, e)

        assert(len(edges) == len(expected))

    def test_collider_5(self):
        '''
        Graph Edges:
          {X1, X2} --> X3; X1 --> X4

        FGES should orient all edges
        '''
        result = run_fges_file("../data/collider_5.txt")
        edges = result['graph'].edges()

        assert_oriented_edge(edges, (0, 2))
        assert_oriented_edge(edges, (1, 2))
        assert_unoriented_edge(edges, (0, 3))

        assert(len(edges) == 4)

    def test_linear_1(self):
        '''
        X1 --> X2 --> X3 --> X4

        FGES should not be able to orient any edges
        '''
        result = run_fges_file("../data/linear_1.txt")
        edges = result['graph'].edges()

        assert_unoriented_edge(edges, (0, 1))
        assert_unoriented_edge(edges, (1, 2))
        assert_unoriented_edge(edges, (2, 3))

    def test_single_edge_1(self):
        '''
        Graph with 10 variables and a single edge.
        '''
        result = run_fges_file("../data/single_edge_1.txt")
        edges = result['graph'].edges()

        assert len(edges) == 2
        assert_unoriented_edge(edges, (0, 1))

    def test_single_edge_2(self):
        '''
        Graph with 50 variables and a single edge.
        '''
        result = run_fges_file("../data/single_edge_2.txt")
        edges = result['graph'].edges()

        assert len(edges) == 2
        assert_unoriented_edge(edges, (0, 1))

    def test_single_edge_3(self):
        '''
        Graph with 100 variables and a single edge.
        '''
        result = run_fges_file("../data/single_edge_3.txt")
        edges = result['graph'].edges()

        print(edges)
        assert len(edges) == 2
        assert_unoriented_edge(edges, (36, 58))

    def test_fifty_edges(self):
        result = run_fges_file("../data/50_edges.txt")
        edges = result['graph'].edges()

        dirs = [e for e in edges if e[::-1] not in edges]
        undirs = [e for e in edges if e[0] < e[1] and e[::-1] in edges]
        assert len(dirs) + len(undirs) == 50


class TestCheckpoints(unittest.TestCase):

    def checkpoint_verify(self, data_file):
        result = run_fges_file(data_file, checkpoint_frequency=1, save_name='test_tmp')
        result2 = FGES.load_checkpoint('test_tmp-checkpoint.pkl').search()
        os.remove("test_tmp-checkpoint.pkl")
        assert set(result['graph'].edges()) == set(result2['graph'].edges())

    def test_checkpoints(self):
        for i in range(1, 6):
            self.checkpoint_verify("../data/collider_{}.txt".format(i))

class RandomFGESTests(unittest.TestCase):
    def test_v_structure(self):
        '''
        Graph Edges:
          X0 --- X1 --> X2 <-- X3
        '''
        g = np.zeros((4, 4))
        g[0, 1] = 1
        g[1, 2] = 1
        g[3, 2] = 1

        d = generate_data(g, [0, 1, 3, 2], 1000)

        result = run_fges_array(d)
        edges = result['graph'].edges()

        assert_unoriented_edge(edges, (0, 1))
        assert_oriented_edge(edges, (1, 2))
        assert_oriented_edge(edges, (3, 2))
        assert len(edges) == 4

    def test_y_structure(self):
        '''
        Graph Edges:
          X0 --> X1 <-- X2
          X1 --> X3
        '''
        g = np.zeros((4, 4))
        g[0, 1] = 1
        g[2, 1] = -1
        g[1, 3] = 2

        d = generate_data(g, [0, 2, 1, 3], 1000)

        result = run_fges_array(d)
        edges = result['graph'].edges()

        print(edges)

        assert_oriented_edge(edges, (0, 1))
        assert_oriented_edge(edges, (2, 1))
        assert_oriented_edge(edges, (1, 3))
        assert len(edges) == 3

    def test_diamond(self):
        '''
        Graph Edges:
          {X0, X1} --> X2 --> {X3, X4} --> X5 <-- X6
        FGES should orient all edges
        '''
        g = np.zeros((7, 7))
        g[0, 2] = 1
        g[1, 2] = 1
        g[2, 3] = 1
        g[2, 4] = 1
        g[3, 5] = 1
        g[4, 5] = 1
        g[6, 5] = 1

        g *= np.random.uniform(0.5, 10, g.shape)
        g *= np.random.choice([-1, 1], g.shape)

        d = generate_data(g, [0, 1, 2, 3, 4, 6, 5], 100000)

        result = run_fges_array(d)
        edges = result['graph'].edges()

        assert_oriented_edge(edges, (0, 2))
        assert_oriented_edge(edges, (1, 2))
        assert_oriented_edge(edges, (2, 3))
        assert_oriented_edge(edges, (2, 4))
        assert_oriented_edge(edges, (3, 5))
        assert_oriented_edge(edges, (4, 5))
        assert_oriented_edge(edges, (6, 5))
        assert len(edges) == 7

if __name__ == "__main__":
    unittest.main()