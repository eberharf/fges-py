
import networkx as nx

from meekrules import MeekRules
from graph_util import *
import numpy as np
import queue

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
    :return: Candidate dag, or None if unsuccessful
    '''

    dag = graph.copy()
    graph_colliders = get_all_collider_triples(graph)
    rules = MeekRules(undirect_unforced_edges=False)

    if detect_cycle(dag):
        return None

    def check_graph(dag, node):
        '''
        Check if a dag is okay.
        :param dag: dag to check
        :param node: node to check for cycles
        :return: dag with implied edges if valid, else None
        '''
        rules.orient_implied(dag)
        new_colliders = get_all_collider_triples(dag)
        if new_colliders == graph_colliders and not detect_cycle_at_node(dag, node):
            return dag
        else:
            return None

    def try_to_solve(pattern):
        '''
        Try to find a fully oriented dag from a pattern using a DFS
        :param pattern: the pattern to try
        :return: dag if successful, else None
        '''
        if pattern is None:
            return None

        edge = get_undir_edge(pattern)

        if edge is None:
            # No more undirected edges
            return pattern

        (node1, node2) = edge
        new_g = pattern.copy()
        if node2 not in get_ancestors(new_g, node1):
            # Orient edge node1 -> node2
            new_g.remove_edge(node2, node1)
            result = try_to_solve(check_graph(new_g, node1))
            if result is not None:
                return result

        new_g = pattern.copy()
        if node1 not in get_ancestors(new_g, node2):
            # Orient edge node2 -> node1
            new_g.remove_edge(node1, node2)
            result = try_to_solve(check_graph(new_g, node2))
            if result is not None:
                return result

        return None

    return try_to_solve(graph)

#TODO: use smarter penalty function than simply a count
def dagFromPatternWithColliders(graph):
    '''
    Construct a DAG from the pattern graph that minimizes the number of new colliders added.
    :param graph: The pattern to use
    :return: DAG
    '''
    pattern = graph.copy()
    original_colliders = get_all_collider_triples(pattern)

    optimal_graph = None
    optimal_penalty = np.inf

    stack = queue.LifoQueue()
    stack.put((pattern, original_colliders, 0))

    while not stack.empty():
        g, colliders, penalty = stack.get()
        if penalty >= optimal_penalty:
            # Branch and Bound
            continue

        found_undirected = False
        # Consider all undirected edges
        for (x, y) in g.edges():
            if not has_undir_edge(g, x, y) or y < x:
                continue

            found_undirected = True

            # Orient x -> y
            g1 = g.copy()
            g1.remove_edge(y, x)

            if not detect_cycle_at_node(g1, x):
                c1 = check_for_colliders(g1, y)
                p1 = penalty + len(c1 - colliders) # set difference
                e1 = g1.edges()
                stack.put((g1, colliders.union(c1), p1))

            # Orient y -> x
            g2 = g.copy()
            g2.remove_edge(x, y)

            if not detect_cycle_at_node(g2, y):
                c2 = check_for_colliders(g2, x)
                p2 = penalty + len(c2 - colliders)  # set difference
                e2 = g2.edges()
                stack.put((g2, colliders.union(c2), p2))

        if not found_undirected:
            # Completely oriented graph
            if penalty < optimal_penalty:
                optimal_graph = g
                optimal_penalty = penalty
                if penalty == 0:
                    break

    return (optimal_graph, optimal_penalty)


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

    return np.array(edge_parameters), np.array(residuals)

def get_covariance_matrix(params, resids):
    '''
    Get the covariance matrix from edge parameters
     (representing a DAG) and the residuals.

    For the equation, see "Causal Mapping of Emotion Networks in the Human Brain" (p. 15)
    '''
    id = np.identity(params.shape[0])
    a = np.linalg.inv(id - params)

    return np.matmul(np.matmul(a, resids), np.transpose(a))