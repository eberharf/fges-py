from SEMScore import *
from fges import *
import unittest

def run_fges(data_file, **kwargs):
    '''
    Run FGES on a data file, returning the resulting graph
    :param data_file: the data file to use
    :return: DiGraph from fges.search()
    '''
    dataset = np.loadtxt(data_file, skiprows = 1)
    score = SEMBicScore(dataset, 2)  # Initialize SEMBic Object
    variables = list(range(len(dataset[0])))
    print("Running FGES on graph with " + str(len(variables)) + " nodes.")
    fges = FGES(variables, score, 10, **kwargs)
    return fges.search()

def assert_unoriented_edge(edges, e):
    # TODO: should this be an `or` and an `and`?
    assert (e in edges) and (e[::-1] in edges)

def assert_oriented_edge(edges, e):
    assert (e in edges) and not (e[::-1] in edges)

class SimpleTests(unittest.TestCase):

    def test_collider_1(self):
        '''
        Test a simple collider X1 -> X3 <- X2
        FGES should obtain the exact graph
        '''
        result = run_fges("../data/collider_1.txt")
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
        result = run_fges("../data/collider_2.txt")
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
        result = run_fges("../data/collider_3.txt")
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
        result = run_fges("../data/collider_4.txt")
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
        result = run_fges("../data/collider_5.txt")
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
        result = run_fges("../data/linear_1.txt")
        edges = result['graph'].edges()

        assert_unoriented_edge(edges, (0, 1))
        assert_unoriented_edge(edges, (1, 2))
        assert_unoriented_edge(edges, (2, 3))

class TestCheckpoints(unittest.TestCase):

    def checkpoint_verify(self, data_file):
        result = run_fges(data_file, checkpoint_frequency=1, save_name='test_tmp')
        result2 = FGES.load_checkpoint('test_tmp-checkpoint.pkl').search()
        os.remove("test_tmp-checkpoint.pkl")
        assert set(result['graph'].edges()) == set(result2['graph'].edges())

    def test_checkpoints(self):
        for i in range(1, 6):
            self.checkpoint_verify("../data/collider_{}.txt".format(i))

if __name__ == "__main__":
    unittest.main()