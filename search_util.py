
import networkx as nx

from meekrules import MeekRules
from graph_util import *

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

