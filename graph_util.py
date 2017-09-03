import queue
import networkx as nx

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

def undir_edge_neighbors(g, x, y):
    return has_undir_edge(g, x, y)

def neighbors(g, x):
    potentialNeighbors = nx.all_neighbors(g, x) #TODO: Check this call
    resulting_neighbors = set({})
    for pNode in potentialNeighbors:
        if undir_edge_neighbors(g, x, pNode):
            resulting_neighbors.add(pNode)

    return resulting_neighbors

def get_na_y_x(g, x, y):
    na_y_x = []
    all_y_neighbors = set(nx.all_neighbors(g, y))

    for z in all_y_neighbors:
        if has_undir_edge(g, z, y):
            if adjacent(g, z, x):
                na_y_x.append(z)

    return na_y_x

def is_clique(g, node_set):
    for node in node_set:
        for other_node in node_set:
            if node != other_node and not adjacent(g, node, other_node):
                return False 
    return True

def get_t_neighbors(g, x, y):
    t = set([])
    all_y_neighbors = set(nx.all_neighbors(g, y))

    for z in all_y_neighbors:
        if has_undir_edge(g, z, y):
            if adjacent(g, z, x):
                continue
            t.add(z)

    return t

def exists_unblocked_semi_directed_path(g, origin, dest, cond_set, bound):
    if bound == -1:
        bound = 1000

    q = queue.Queue()
    v = set()
    q.put(origin)
    v.add(origin)

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

            if c in cond_set:
                continue 

            if c == dest:
                return True 

            if not c in v:
                v.add(c)
                q.put(c)

                if e == None: 
                    e = c 
    return False
