import networkx as nx
import itertools
import graph_util
from sortedcontainers import SortedListWithKey

class FGES:
    """
    Python FGES implementation, heavily inspired by tetrad 
    https://github.com/cmu-phil/tetrad
    """

    def __init__(self, variables, score, maxDeg):
        self.top_graphs = []
        self.variables = variables
        self.score = score
        self.sorted_arrows = SortedListWithKey(key=lambda val: -val.bump)
        self.arrow_dict = {}
        self.arrow_index = 0
        self.total_score = 0
        # Only needed for their `heuristic speedup`, it tells
        # you if two edges even have an effect on each other
        # the way we use this is effect_edges_graph[node] gives you 
        # an iterable of nodes {w_1, w_2, w_3...} where node and 
        # w_i have a non-zero total effect
        self.effect_edges_graph = {}
        self.max_degree = maxDeg
        self.cycle_bound = -1
        self.stored_neighbors = {}
        self.graph = None


    def search(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.variables)
        # for now, there is no knowledge and faithfulness is assumed

        #TODO: self.addRequiredEdges()

        self.initializeForwardEdgesFromEmptyGraph()



    def fes(self):
        while(len(self.sorted_arrows) > 0):
            max_bump_arrow = self.sorted_arrows.pop(0)
            x = max_bump_arrow.a
            y = max_bump_arrow.b

            if graph_util.adjacent(self.graph, x, y):
                continue

            naYX = graph_util.get_na_y_x(self.graph, x, y)

            # TODO: max degree checks

            if max_bump_arrow.naYX != naYX:
                continue

            if not graph_util.get_t_neighbors(self.graph, x, y).issuperset(max_bump_arrow.hOrT):
                continue

            if not self.valid_insert(x, y, max_bump_arrow.hOrT, naYX):
                continue

            T = max_bump_arrow.hOrT
            bump = max_bump_arrow.bump

            #TODO: Insert should return a bool that we check here
            self.insert(self.graph, x, y, T)

            self.total_score += bump

            visited_nodes = self.reapply_orientation(x, y, None)
            to_process = set({})

            # check whether the (undirected) neighbors of each node in 
            # visited_nodes changed compared to stored neighbors
            for node in visited_nodes:
                # gets undirected neighbors 
                new_neighbors = graph_util.neighbors(self.graph, node)
                stored_neighbors = self.stored_neighbors[node]
                if stored_neighbors != new_neighbors:
                    to_process.add(node)

            to_process.add(x)
            to_process.add(y)

            #TODO: storeGraph()

            self.reevaluate_forward(to_process, max_bump_arrow)


    def reevaluate_forward(self, to_process, arrow):
        #TODO: This leverages effect_edges_graph
        for node in to_process:
            nzero_effect_nodes = self.effect_edges_graph[node]

            for w in nzero_effect_nodes:
                if w == node: 
                    continue 
                if graph_util.adjacent(self.graph, node, w):
                    self.clear_arrow(w, node)
                    # self.calcArrowsForward(w, node)

    def reapply_orientation(self, x, y, new_arrows):
        #TODO: Not sure what new_arrows does here, since it is passed as null
        #in fes(), but it should (for some reason) be a list of nodes (not arrows!)
        to_process = set([x, y])
        if new_arrows is not None:
            to_process.update(new_arrows)

    def meekOrientRestricted(self):
        #TODO: This seems like it uses knowledge to re-orient some edges?
        pass


    def valid_insert(self, x, y, T, naYX):
        union = T
        union.append(naYX)
        return graph_util.is_clique(self.graph, union) and \
            not graph_util.exists_unblocked_semi_directed_path(self.graph, y, x, union, self.cycle_bound)

    #TODO: initialzeForwardEdgesFromExistingGraph

    def initializeForwardEdgesFromEmptyGraph(self):

        neighbors = []
        #TODO: Parallelize this in chunks, as the java code does this with
        #the task framework
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                bump = self.score.localScoreDiff(j, i)

                if bump > 0:
                    #TODO:
                    #The java code here keeps track of an edgeEffectsGraph
                    #where X--Y means that X and Y have a non-zero effect
                    #on each other (?)
                    pass
                    # self.addArrow(a, b, [], set([]), bump)
                    # self.addArrow(b, a, [], set([]), bump)


    def insert(self, graph, x, y, T):
        """ T is a subset of the neighbors of Y that are not adjacent to
        (connected by a directed or undirected edge) to X, this should
        connect X -> Y and for t \in T, direct T -> Y if it's not already
        directed

        Definition 12

        """
        graph.add_edge(x, y)

        for node in T:
            graph_util.undir_to_dir(graph, node, y)


    def delete(self, graph, x, y, H):
        # self.remove_all_edges(graph, x, y)

        for node in H:
            graph_util.undir_to_dir(graph, y, node)
            graph_util.undir_to_dir(graph, x, node)

    def addArrow(self, a, b, naYX, hOrT, bump):
        a = Arrow(a, b, naYX, hOrT, bump, self.arrow_index)
        self.sorted_arrows.add(a)

        pair = (a, b)
        if self.arrow_dict.get(pair) is None:
            self.arrow_dict[pair] = a

        self.arrow_dict[pair].append(a)

        self.arrow_index += 1

    def clear_arrow(self, a, b):
        pair = (a, b)
        if self.arrow_dict.get(pair) is not None:
            self.arrow_dict[pair] = None 

    def calculateArrowsForward(self, a, b):
        if b not in self.effect_edges_graph[a]:
            return 

        if self.stored_neighbors.get(b) is None: 
            self.stored_neighbors[b] = set({a})
        else:
            self.stored_neighbors[b].add(a)

        na_y_x = graph_util.get_na_y_x(self.graph, a, b)
        _na_y_x= list(na_y_x)

        if graph_util.is_clique(self.graph, na_y_x):
            return 
        
        t_neighbors = graph_util.get_t_neighbors(self.graph, a, b)
        len_T = len(t_neighbors)

        previous_cliques = set() # set of sets of nodes
        previous_cliques.add(set()) 

        new_cliques = set() # set of sets of nodes

        for i in range(len_T):

            choices = itertools.combinations(range(0, len_T), i)

            for choice in choices: 
                diff = set([_na_y_x[i] for c in choice])
                h = set(na_y_x)
                h = h.difference(diff)

                # bump = delete_eval(a, b, diff, naYx)
                bump = 0.0

                if bump > 0.0:
                    self.addArrow(a, b, na_y_x, h, bump)




class Arrow:

    def __init__(self, a, b, naYX, hOrT, bump, arrow_index):
        self.a = a
        self.b = b
        self.na_y_x = naYX
        self.h_or_t = hOrT
        self.bump = bump
        self.index = arrow_index
