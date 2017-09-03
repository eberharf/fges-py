import networkx as nx
import itertools
from graph_util import *
from sortedcontainers import SortedListWithKey

class FGES:

    def __init__(self, variables, score, maxDeg):
        self.top_graphs = []
        self.variables = variables
        self.score = score
        self.sorted_arrows = SortedListWithKey(key=lambda val: -val.bump)
        self.arrow_dict = {}
        self.arrow_index = 0
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



    def fes():
        while(len(self.sorted_arrows) > 0):
            maxBumpArrow = self.sorted_arrows.pop(0)
            x = maxBumpArrow.a
            y = maxBumpArrow.b

            if adjacent(self.graph, x, y):
                continue

            naYX = getNaYX(g, x, y)

            # TODO: max degree checks

            if maxBumpArrow.naYX != naYX:
                continue

            if not getTNeighbors(g, x, y).issuperset(maxBumpArrow.hOrT):
                continue

            if not validInsert(x, y, maxBumpArrow.hOrT, naYX):
                continue

            T = maxBumpArrow.hOrT
            bump = maxBumpArrow.bump

            #TODO: Insert should return a bool that we check here
            self.insert(self.graph, x, y, T)

            self.totalScore += bump

            visitedNodes = reapplyOrientation(x, y, null)
            toProcess = set({})

            # check whether the (undirected) neighbors of each node in 
            # visitedNodes changed compared to stored neighbors
            for node in visitedNodes:
                # gets undirected neighbors 
                newNeighbors = neighbors(self.graph, node)
                stored_neighbors = self.stored_neighbors[node]
                if stored_neighbors != newNeighbors:
                    toProcess.add(node)

            toProcess.add(x)
            toProcess.add(y)

            #TODO: storeGraph()

            reevaluateForward(toProcess, maxBumpArrow)


    def reevaluateForward(self, toProcess, arrow):
        #TODO: This leverages effect_edges_graph
        for node in toProcess:
            nzero_effect_nodes = effect_edges_graph[x]

            for w in nzero_effect_nodes:
                if w == node: 
                    continue 
                if adjacent(self.graph, node, w):
                    self.clearArrow(w, node)
                    self.calcArrowsForward(w, node)

    def reapplyOrientation(self, x, y, newArrows):
        #TODO: Not sure what newArrows does here, since it is passed as null
        #in fes(), but it should (for some reason) be a list of nodes (not arrows!)
        toProcess = set([x, y])
        if newArrows != null:
            toProcess.append(newArrows)

    def meekOrientRestricted(self):
        #TODO: This seems like it uses knowledge to re-orient some edges?
        pass


    def validInsert(x, y, T, naYX):
        union = T
        union.append(naYX)
        return is_clique(g, union) and not existsUnblockedSemiDirectedPath(y, x, union, self.cycle_bound)

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
                    self.addArrow(a, b, [], set([]), bump)
                    self.addArrow(b, a, [], set([]), bump)


    def insert(self, graph, x, y, T):
        """ T is a subset of the neighbors of Y that are not adjacent to
        (connected by a directed or undirected edge) to X, this should
        connect X -> Y and for t \in T, direct T -> Y if it's not already
        directed

        Definition 12

        """
        graph.add_edge(x, y)

        for node in T:
            undir_to_dir(graph, node, y)


    def delete(self, graph, x, y, H):
        self.remove_all_edges(graph, x, y)

        for node in H:
            undir_to_dir(graph, y, node)
            undir_to_dir(graph, x, node)

    def addArrow(self, a, b, naYX, hOrT, bump):
        a = Arrow(a, b, naYX, hOrT, bump, self.arrow_index)
        self.sorted_arrows.add(a)

        pair = (a, b)
        if self.arrow_dict.get(pair) is None:
            self.arrow_dict[pair] = arrow

        self.arrow_dict[pair].append(arrow)

        self.arrow_index += 1

    def clearArrow(self, a, b):
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

        naYX = getNaYX(self.graph, a, b)
        _naYX = list(naYX)

        if is_clique(self.graph, naYX):
            return 
        
        tNeighbors = getTNeighbors(self.graph, a, b)
        lenT = len(tNeighbors)

        previousCliques = set() # set of sets of nodes
        previousCliques.add(set()) 

        newCliques = set() # set of sets of nodes

        for i in range(lenT):

            choices = itertools.combinations(range(0, lenT), i)

            for choice in choices: 
                diff = set([_naYX[i] for c in choice])
                h = set(_naYX)
                h = h.difference(diff)

                bump = deleteEval(a, b, diff, naYx)

                if bump > 0.0:
                    self.addArrow(a, b, naYX, h, bump)




class Arrow:

    def __init__(a, b, naYX, hOrT, bump, arrow_index):
        self.a = a
        self.b = b
        self.naYX = naYX
        self.hOrT = hOrT
        self.bump = bump
        self.index = arrow_index
