import networkx as nx
import queue

def add_undir_edge(g, x, y):
    g.add_edge(x, y)
    g.add_edge(y, x)

def undir_to_dir(g, x, y):
    """ Keep only x-> y """
    g.remove_edge(y, x)
    assert(g.has_edge(x, y))

def has_undir_edge(g, x, y):
    """ undir edge is x <-> y """
    if g.has_edge(x, y) and g.has_edge(y, x):
        return True 
    return False

def traverseSemiDirected(g, x, y):
    if has_undir_edge(g, x, y):
        return y
    if g.has_edge(x, y):
        return y 
    return None

def remove_all_edges(g, x, y):
    g.remove_edge(x, y)
    g.remove_edge(y, x)

def adjacent(g, x, y):
    """ Connected by an undirected or directed edge """
    if g.has_edge(x, y) or g.has_edge(y, x):
        return True 
    return False

def neighbors(g, x, y):
    return has_undir_edge(g, x, y)

def neighbors(g, x):
    potentialNeighbors = nx.all_neighbors(g, t)
    neighbors = set({})
    for pNode in potentialNeighbors:
        if neighbors(g, x, pNode):
            neighbors.add(pNode)

    return neighbors

def getNaYX(g, x, y):
    nayx = []
    all_y_neighbors = set(nx.all_neighbors(g, y))

    for z in all_y_neighbors:
        if has_undir_edge(g, z, y):
            if adjacent(g, z, x):
                nayx.append(z)

    return nayx

def is_clique(g, nodeSet):
    for node in nodeSet:
        for otherNode in nodeSet:
            if node != otherNode and not adjacent(g, node, otherNode):
                return False 
    return True

def getTNeighbors(g, x, y):
    t = set([])
    all_y_neighbors = set(nx.all_neighbors(g, y))

    for z in all_y_neighbors:
        if has_undir_edge(g, z, y):
            if adjacent(g, z, x):
                continue
            t.append(z)

    return t

def existsUnblockedSemiDirectedPath(g, origin, dest, condSet, bound):
    if bound == -1:
        bound = 1000

    q = Queue()
    v = set()
    q.put(origin)
    v.append(origin)

    e = None 
    distance = 0

    while not q.empty():
        t = q.get()
        if t == dest:
            return True 

        if e == t:
            e = None 
            distance += 1
            if distance > bound:
                return False 

        for u in set(nx.all_neighbors(g, t)):
            c = traverseSemiDirected(g, t, u)
            if c is None:
                continue 

            if condSet.contains(c):
                continue 

            if c == dest:
                return True 

            if not v.contains(c):
                v.add(c)
                q.put(c)

                if e == None: 
                    e = c 
    return False
