import networkx as nx
import itertools
import graph_util
from sortedcontainers import SortedListWithKey
from meekrules import MeekRules


class FGES:
    """
    Python FGES implementation, heavily inspired by tetrad 
    https://github.com/cmu-phil/tetrad
    """

    def __init__(self, variables, score, maxDeg):
        self.top_graphs = []
        # List<Node> --> list of variables, in order
        self.variables = variables
        self.node_dict = {} #hash_indices
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
        self.removed_edges = set()

    def search(self):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.variables)
        # for now, there is no knowledge and faithfulness is assumed

        #TODO: self.addRequiredEdges()

        self.initialize_forward_edges_from_empty_graph()

        self.mode = "heuristic"
        fes()
        bes()

        self.mode = "covernoncolliders"
        fes()
        bes()

        return self.graph 
        

    def fes(self):
        while(len(self.sorted_arrows) > 0):
            max_bump_arrow = self.sorted_arrows.pop(0)
            x = max_bump_arrow.a
            y = max_bump_arrow.b

            if graph_util.adjacent(self.graph, x, y):
                continue

            na_y_x = graph_util.get_na_y_x(self.graph, x, y)

            # TODO: max degree checks

            if max_bump_arrow.na_y_x != na_y_x:
                continue

            if not graph_util.get_t_neighbors(self.graph, x, y).issuperset(max_bump_arrow.hOrT):
                continue

            if not self.valid_insert(x, y, max_bump_arrow.hOrT, na_y_x):
                continue

            T = max_bump_arrow.hOrT
            bump = max_bump_arrow.bump

            # TODO: Insert should return a bool that we check here
            inserted = self.insert(self.graph, x, y, T)
            if (not inserted):
                continue

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
    
    def bes(self):
        self.sorted_arrows = SortedListWithKey(key=lambda val: -val.bump)
        self.arrow_dict = {} 
        self.stored_neighbors = {}

        self.initialize_arrows_backwards()

        while len(sorted_arrows) > 0:
            arrow = sorted_arrows.pop(0)
            x = arrow.a
            y = arrow.b 

            if (not arrow.na_y_x is graph_util.get_na_y_x(self.graph, x, y)):
                continue
            
            if (not graph_util.adjacent(self.graph, x, y)):
                continue 
            
            if (graph_util.has_dir_edge(self.graph, y, x)): # edge points towards x
                continue 
            
            diff = set(arrow.na_y_x)
            diff = diff - arrow.h_or_t

            if (not self.valid_delete(x, y, arrow.h_or_t, arrow.na_y_x))
                continue 
            
            H = arrow.h_or_t 
            bump = arrow

            self.delete(self.graph, x, y, H)

            meek_rules = MeekRules()
            meek_rules.orient_implied_subset(self.graph, set(x))
            meek_rules.orient_implied_subset(self.graph, set(y))

            self.total_score += bump 
            self.clear_arrow(x, y)

            visited = self.reapply_orientation(x, y, H)

            to_process = set()

            for node in visited: 
                neighbors = graph_util.neighbors(self.graph, node)
                str_neighbors = self.stored_neighbors[node]

                if stored_neighbors != new_neighbors:
                    to_process.update(node)
            
            to_process.add(x)
            to_process.add(y)
            to_process.update(graph_util.get_common_adjacents(self.graph, x, y))

            #TODO: Store graph
            self.reevaluate_backward(to_process)



        
    def initialize_arrows_backwards(self):
        for (node_1, node_2) in self.graph.edges():
            
            self.clear_arrow(node_1, node_2)
            self.clear_arrow(node_2, node_1)
            
            #TODO: Confirm this is equivalent
            self.calculate_arrows_backward(node_1, node_2)

            self.stored_neighbors[node_1] = graph_util.neighbors(self.graph, node_1)
            self.stored_neighbors[node_2] = graph_util.neighbors(self.graph, node_2)

    def calculate_arrows_backward(self, a, b):
        na_y_x = graph_util.get_na_y_x(self.graph, a, b)
        _na_y_x = list(na_y_x)
        _depth  = len(_na_y_x)

        for i in range(_depth):
            choices = itertools.combinations(range(0, _depth), i)
            for choice in choices:
                diff = set([_na_y_x[k] for k in choice])
                h = set(_na_y_x)
                h = h - diff 

                bump = self.delete_eval(a, b, diff, na_y_x, self.node_dict)

                if bump > 0:
                    self.add_arrow(a, b, na_y_x, h, bump)
            
    
    def clear_arrow(self, x, y):
        del self.arrow_dict[(x, y)]
        
    def delete_eval(self, x, y, diff, na_y_x, node_dict):
        a = set(diff)
        a.update(graph_util.get_parents(self.graph, y))
        a = a - set(x)
        return -1 * self.score_graph_change(y, a, x, node_dict)

    def reevaluate_forward(self, to_process, arrow):
        # TODO: This leverages effect_edges_graph
        for node in to_process:
            if self.mode == "heuristic":
                nzero_effect_nodes = self.effect_edges_graph[node]
            elif self.mode == "covernoncolliders":
                g = set()
                for n in graph_util.adjacent_nodes(self.graph, node):
                    for m in graph_util.adjacent_nodes(self.graph, n):
                        if graph_util.adjacent(self.graph, n, m):
                            continue 
                        
                        if graph_util.is_def_collider(self.graph, m, n, node):
                            continue 

                        g.update(m)
                
                nzero_effect_nodes = list(g)

            for w in nzero_effect_nodes:
                if w == node:
                    continue
                if graph_util.adjacent(self.graph, node, w):
                    self.clear_arrow(w, node)
                    self.calculate_arrows_forward(w, node)
    
    def reevaluate_backward(self, to_process):
        for node in to_process:
            self.stored_neighbors[node] = graph_util.neighbors(node)
            adjacent_nodes = graph_util.adjacent_nodes(node)

            for adj_node in adjacent_nodes:
                if (graph_util.has_dir_edge(self.graph, adj_node, node)):
                    self.clear_arrow(adj_node, node)
                    self.clear_arrow(node, adj_node)

                    self.calculate_arrows_backward(adj_node, node)
                else if graph_util.has_undir_edge(self.graph, adj_node, node):
                    self.clear_arrow(adj_node, node)
                    self.clear_arrow(node, adj_node)
                    self.calculate_arrows_backward(adj_node, node)
                    self.calculate_arrows_backward(node, adj_node)


    def knowledge(self):
        return None

    def reapply_orientation(self, x, y, new_arrows):
        # TODO: Not sure what new_arrows does here, since it is passed as null
        # in fes(), but it should (for some reason) be a list of nodes (not arrows!)
        to_process = set([x, y])
        if new_arrows is not None:
            to_process.update(new_arrows)

        return self.meek_orient_restricted(self.graph, to_process, self.knowledge())

    def meek_orient_restricted(self, graph, nodes, knowledge):
        # Runs meek rules on the changed adjacencies
        meek_rules = MeekRules(undirect_unforced_edges=True)
        meek_rules.orient_implied_subset(graph, nodes)
        return meek_rules.get_visited()

    def valid_insert(self, x, y, T, na_y_x):
        union = T
        union.append(na_y_x)
        return graph_util.is_clique(self.graph, union) and \
            not graph_util.exists_unblocked_semi_directed_path(
                self.graph, y, x, union, self.cycle_bound)
                
    def valid_delete(self, x, y, H, na_y_x):
        #TODO Knowledge
        diff = set(na_y_x)
        diff = diff - H
        return graph_util.is_clique(self.graph, diff)

    #TODO: initialzeForwardEdgesFromExistingGraph

    def initialize_two_step_edges(self, nodes):
        for node in nodes:

            g = set()

            for n in graph_util.adjacent_nodes(self.graph, node):
                for m in graph_util.adjacent_nodes(self.graph, n):

                    if node == m:
                        continue

                    if graph_util.adjacent(self.graph, node, m):
                        continue 

                    if graph_util.is_def_collider(self.graph, m, n, node):
                        continue 

                    g.update(m)
            
            for x in g:
                assert(x !== node)
                #TODO: Knowledge

                #TODO: Adjacencies

                if (x, node) in self.removed_edges:
                    continue 

                self.calculate_arrows_forward(x, node)


    def initialize_forward_edges_from_empty_graph(self):

        neighbors = []
        # TODO: Parallelize this in chunks, as the java code does this with
        # the task framework
        for i in range(len(self.variables)):
            for j in range(i + 1, len(self.variables)):
                bump = self.score.local_score_diff(j, i)

                if bump > 0:
                    # TODO:
                    # The java code here keeps track of an edgeEffectsGraph
                    # where X--Y means that X and Y have a non-zero effect
                    # on each other (?)
                    # pass
                    parent_node = self.variables[j]
                    child_node = self.variables[i]
                    self.add_arrow(parent_node, child_node, [], set([]), bump)
                    self.add_arrow(child_node, parent_node, [], set([]), bump)

        print("Initialized forward edges from empty graph")

    def insert_eval(self, x, y, T, na_y_x, node_dict):
        # node_dict is supposed to be a map from node --> index
        assert(x is not y)
        _na_y_x = set(na_y_x)
        _na_y_x.update(T)
        _na_y_x.update(graph_util.get_parents(self.graph, y))
        return self.score_graph_change(y, _na_y_x, x, node_dict)
    
    def delete_eval(self, x, y, diff, node_dict):
        # node_dict is supposed to be a map from node --> index
        _diff = set(diff)
        _diff.update(graph_util.get_parents(self.graph, y))
        _diff.remove(x)
        return -1 * self.score_graph_change(y, diff, x, node_dict)

    def insert(self, graph, x, y, T):
        """ T is a subset of the neighbors of Y that are not adjacent to
        (connected by a directed or undirected edge) to X, this should
        connect X -> Y and for t \in T, direct T -> Y if it's not already
        directed

        Definition 12

        """
        if graph_util.adjacent(graph, x, y):
            return False

        # TODO Bound Graph

        # Adds directed edge
        graph.add_edge(x, y)

        # TODO print number of edges

        for node in T:
            graph_util.undir_to_dir(graph, node, y)

        return True

    def delete(self, graph, x, y, H):
        # self.remove_all_edges(graph, x, y)

        for node in H:
            graph_util.undir_to_dir(graph, y, node)
            graph_util.undir_to_dir(graph, x, node)
            self.removed_edges.add((x, y))

    def add_arrow(self, a, b, na_y_x, h_or_t, bump):
        a = Arrow(a, b, na_y_x, h_or_t, bump, self.arrow_index)
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

    def score_graph_change(self, y, parents, x, node_dict):
        # node_dict is supposed to be a map from node --> index
        assert (x is not y)
        assert (y not in parents)
        y_index = node_dict[y]

        parent_indices = list()
        for parent_node in parents:
            parent_indices.append(node_dict[parent_node])

        return self.score.local_score_diff_parents(node_dict[x], y_index, parent_indices)

    def calculate_arrows_forward(self, a, b):
        if b not in self.effect_edges_graph[a]:
            return

        if self.stored_neighbors.get(b) is None:
            self.stored_neighbors[b] = set({a})
        else:
            self.stored_neighbors[b].add(a)

        na_y_x = graph_util.get_na_y_x(self.graph, a, b)
        _na_y_x = list(na_y_x)

        if graph_util.is_clique(self.graph, na_y_x):
            return

        t_neighbors = graph_util.get_t_neighbors(self.graph, a, b)
        len_T = len(t_neighbors)


        def outer_loop():
            previous_cliques = set()  # set of sets of nodes
            previous_cliques.add(set())
            new_cliques = set()  # set of sets of nodes
            for i in range(len_T):

                # TODO: Check that this does the same thing as ChoiceGenerator
                choices = itertools.combinations(range(0, len_T), i)

                for choice in choices:
                    T = set([t_neighbors[k] for k in choice])
                    union = set(na_y_x)
                    union.update(T)

                    found_a_previous_clique = False

                    for clique in previous_cliques:
                        # basically if clique is a subset of union
                        if union >= clique:
                            found_a_previous_clique = True
                            break

                    if not found_a_previous_clique:
                        # Break out of the outer for loop
                        return

                    if not graph_util.is_clique(self.graph, union):
                        continue

                    new_cliques.add(union)

                    bump = self.insert_eval(a, b, T, na_y_x, self.node_dict)

                    if bump > 0:
                        self.add_arrow(a, b, na_y_x, T, bump)
            
                previous_cliques = new_cliques
                new_cliques = set()
        
        outer_loop()

class Arrow:

    def __init__(self, a, b, na_y_x, hOrT, bump, arrow_index):
        self.a = a
        self.b = b
        self.na_y_x = na_y_x
        self.h_or_t = hOrT
        self.bump = bump
        self.index = arrow_index
