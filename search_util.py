
import networkx as nx

from meekrules import MeekRules
from graph_util import *
import numpy as np

def get_undir_edge(g):
    '''Find an undirected edge in Graph g, or None if none exist'''
    for (x, y) in g.edges():
        if has_undir_edge(g, x, y):
            return (x, y)
    return None

def dagFromPattern(graph):
    '''
    Construct a specific graph from an equivalence class
    :param graph: Equivalence class (i.e. DAG with unoriented edges)
    :return: Candidate member (ie. DAG with no unoriented edges, a subset of input graph)
    '''

    dag = graph.copy()
    graph_colliders = get_all_collider_triples(graph)
    rules = MeekRules(undirect_unforced_edges=False)

    # Tuples of (graph, last edge oriented, orientation status)
    # Orientation status: 0 = unoriented, 1 = forward, 2 = backward
    choices = []
    choices.append((dag, get_undir_edge(dag), 0))

    while len(choices) > 0:
        (g, edge, status) = choices.pop(-1)

        if edge is None:
            # No more undirected edges
            return g

        (node1, node2) = edge
        new_g = g.copy()

        if (status == 0) and (node2 not in get_ancestors(new_g, node1)):
            # oreint node1 -> node2
            new_g.remove_edge(node2, node1)
            new_status = 1
        elif (status <= 1) and (node1 not in get_ancestors(new_g, node2)):
            # orient node2 -> node1
            new_g.remove_edge(node1, node2)
            if not new_g.has_edge(node2, node1):
                new_g.add_edge(node2, node1)
            new_status = 2
        else:
            # Go back to prior decision
            continue

        rules.orient_implied(new_g)

        new_colliders = get_all_collider_triples(new_g)
        if new_colliders == graph_colliders:
            # Commit to orientation for time being
            choices.append((new_g, edge, new_status))
            choices.append((new_g, get_undir_edge(new_g), 0))
        else:
            # Try other orientation
            choices.append((g, edge, new_status))

def mean_shift_data(data):
    '''Shift all variables in a dataset to have mean zero'''
    return data - np.mean(data, axis=0)

def estimate_parameters(dag, data):
    '''
    Estimate the parameters of a DAG to fit the data.
    :return: matrix of edge coefficients, and diagonal matrix of residuals
    '''

    assert get_undir_edge(dag) is None

    data = mean_shift_data(data)
    num_nodes = len(dag.nodes())

    edge_parameters = np.zeros((num_nodes, num_nodes))
    residuals = np.zeros((num_nodes, num_nodes))

    for j in range(num_nodes):
        inbound_nodes = [i for i in range(num_nodes) if has_dir_edge(dag, i, j)]

        if len(inbound_nodes) == 0:
            residuals[j, j] = np.var(data[:, j])
            continue

        assert j not in inbound_nodes

        a = data[:, inbound_nodes]
        b = data[:, j]

        params, r, _, _ = np.linalg.lstsq(a, b)

        residuals[j, j] = r / (data.shape[0] - 1)

        for i in range(len(inbound_nodes)):
            edge_parameters[j, inbound_nodes[i]] = params[i]
            # v = edge_parameters * v + e

    return edge_parameters, residuals

def get_covariance_matrix(params, resids):
    '''
    Get the covariance matrix from edge parameters
     (representing a DAG) and the residuals.

    For the equation, see "Causal Mapping of Emotion Networks in the Human Brain" (p. 15)
    '''
    id = np.identity(params.shape[0])
    a = np.linalg.inv(id - params)

    return np.matmul(np.matmul(a, resids), np.transpose(a))