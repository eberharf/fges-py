import graph_util
import itertools


class MeekRules:
    def __init__(self, undirect_unforced_edges=True, knowledge=None):
        # Unforced parents should be undirected before orienting
        self.undirect_unforced_edges = undirect_unforced_edges
        self.init_nodes = []
        self.node_subset = {}
        self.visited = set()
        self.direct_stack = []
        self.oriented = set()
        self.knowledge=knowledge

    def orient_implied_subset(self, graph, node_subset):
        self.node_subset = node_subset
        self.visited.update(node_subset)
        self.orient_using_meek_rules_locally(None, graph)

    def orient_implied(self, graph):
        self.orient_implied_subset(graph, graph.nodes())

    def orient_using_meek_rules_locally(self, knowledge, graph):
        """Orient graph using the four Meek rules"""
        if (self.undirect_unforced_edges):
            for node in self.node_subset:
                self.undirect_unforced_edges_func(node, graph)
                self.direct_stack.extend(
                    graph_util.adjacent_nodes(graph, node))


        # TODO: Combine loops
        for node in self.node_subset:
            self.run_meek_rules(node, graph, knowledge)
        if self.direct_stack != []:
            last_node = self.direct_stack.pop()
        else:
            last_node = None

        while last_node is not None:
            # print(last_node)
            if (self.undirect_unforced_edges):
                self.undirect_unforced_edges_func(last_node, graph)

            self.run_meek_rules(last_node, graph, knowledge)
            # print("past run_meek_rules")
            if (len(self.direct_stack) > 0):
                last_node = self.direct_stack.pop()
            else:
                last_node = None

    def undirect_unforced_edges_func(self, node, graph):
        """Removes directed edges that are not forced by an unshielded collider about node"""
        node_parents = graph_util.get_parents(graph, node)
        parents_to_undirect = set(node_parents)

        # Find any unshielded colliders in node_parents, and orient them
        for (p1, p2) in itertools.combinations(node_parents, 2):
            if not graph_util.adjacent(graph, p1, p2):
                # Have an unshielded collider p1 -> node <- p2, which forces orientation
                self.oriented.update([(p1, node), (p2, node)])
                parents_to_undirect.difference_update([p1, p2])

        did_unorient = False

        for parent in parents_to_undirect:
            if self.knowledge is not None:
                must_orient = self.knowledge.is_required(parent, node) or \
                              self.knowledge.is_forbidden(node, parent)
            else:
                must_orient = False

            if not (parent, node) in self.oriented and not must_orient:
                # Undirect parent -> node
                graph_util.remove_dir_edge(graph, parent, node)
                graph_util.add_undir_edge(graph, parent, node)
                self.visited.add(node)
                self.visited.add(parent)
                print(f"unorienting {parent} -> {node}")
                did_unorient = True

        if did_unorient:
            for adjacent in graph_util.adjacent_nodes(graph, node):
                self.direct_stack.append(adjacent)

            self.direct_stack.append(node)

    def run_meek_rules(self, node, graph, knowledge):
        self.run_meek_rule_one(node, graph, knowledge)
        self.run_meek_rule_two(node, graph, knowledge)
        self.run_meek_rule_three(node, graph, knowledge)
        self.run_meek_rule_four(node, graph, knowledge)

    def run_meek_rule_one(self, node, graph, knowledge):
        """
                Meek's rule R1: if a-->b, b---c, and a not adj to c, then a-->c
                """
        # print("Running meek rule one", node)
        adjacencies = graph_util.adjacent_nodes(graph, node)
        if len(adjacencies) < 2:
            return
        all_combinations = itertools.combinations(
            range(0, len(adjacencies)), 2)
        for (index_one, index_two) in all_combinations:
            node_a = adjacencies[index_one]
            node_c = adjacencies[index_two]

            # TODO: Parallelize these flipped versions?
            self.r1_helper(node_a, node, node_c, graph, knowledge)
            self.r1_helper(node_c, node, node_a, graph, knowledge)

    def r1_helper(self, node_a, node_b, node_c, graph, knowledge):
        if ((not graph_util.adjacent(graph, node_a, node_c)) and (graph_util.has_dir_edge(graph, node_a, node_b) or (node_a, node_b) in self.oriented) and graph_util.has_undir_edge(graph, node_b, node_c)):
            if (not graph_util.is_unshielded_non_collider(graph, node_a, node_b, node_c)):
                return

            if self.is_arrowpoint_allowed(graph, node_b, node_c, knowledge):
                #print("R1: " + str(node_b) + " " + str(node_c))
                if (node_a, node_c) not in self.oriented and (node_c, node_a) not in self.oriented and \
                    (node_b, node_c) not in self.oriented and (node_c, node_b) not in self.oriented:
                        self.direct(node_b, node_c, graph)

    def direct(self, node_1, node_2, graph):
        # print("Int Directing " + str(node_1) + " " + str(node_2))
        if self.knowledge is not None:
            if self.knowledge.is_forbidden(node_1, node_2):
                return

        graph_util.remove_dir_edge(graph, node_1, node_2)
        graph_util.remove_dir_edge(graph, node_2, node_1)
        graph_util.add_dir_edge(graph, node_1, node_2)
        self.visited.update([node_1, node_2])
        # node_1 -> node_2 edge
        if (node_1, node_2) not in self.oriented and (
        node_2, node_1) not in self.oriented:
            self.oriented.add((node_1, node_2))
            self.direct_stack.append(node_2)

    def run_meek_rule_two(self, node_b, graph, knowledge):
        # print("Running meek rule two", node_b)
        adjacencies = graph_util.adjacent_nodes(graph, node_b)
        if len(adjacencies) < 2:
            return
        all_combinations = itertools.combinations(
            range(0, len(adjacencies)), 2)
        for (index_one, index_two) in all_combinations:
            node_a = adjacencies[index_one]
            node_c = adjacencies[index_two]
            self.r2_helper(node_a, node_b, node_c, graph, knowledge)
            self.r2_helper(node_b, node_a, node_c, graph, knowledge)
            self.r2_helper(node_a, node_c, node_b, graph, knowledge)
            self.r2_helper(node_c, node_a, node_b, graph, knowledge)

    def r2_helper(self, a, b, c, graph, knowledge):
        if graph_util.has_dir_edge(graph, a, b) and \
                graph_util.has_dir_edge(graph, b, c) and \
                graph_util.has_dir_edge(graph, a, c):
            if self.is_arrowpoint_allowed(graph, a, c, knowledge):
                self.direct(a, c, graph)

    def run_meek_rule_three(self, node, graph, knowledge):
        '''
        A --- B, A --- X, A --- C, B --> X <-- C, B -/- C => A --> X
        The parameter node = X
        '''
        # print("Running meek rule three", node)
        adjacencies = graph_util.adjacent_nodes(graph, node)

        if len(adjacencies) < 3:
            return

        for node_a in adjacencies:
            if graph_util.has_undir_edge(graph, node, node_a):
                copy_adjacencies = [a for a in adjacencies if a != node_a]
                all_combinations = itertools.combinations(copy_adjacencies, 2)

                for node_b, node_c in all_combinations:
                    if graph_util.is_kite(graph, node, node_a, node_b, node_c) and \
                            self.is_arrowpoint_allowed(graph, node_a, node, None) and \
                            graph_util.is_unshielded_non_collider(graph, node_c, node_a, node_b):
                        # print("R3: " + str(node_a) + " " + str(node))
                        self.direct(node_a, node, graph)

    def run_meek_rule_four(self, a, graph, knowledge):
        if knowledge is None:
            return

        adjacent_nodes = graph_util.adjacent_nodes(graph, a)
        for c in adjacent_nodes:
            adjacent_nodes.remove(c)
            all_combinations = itertools.combinations(adjacent_nodes, 2)
            for b, d in all_combinations:
                if not graph_util.adjacent(graph, a, b) and \
                   graph_util.adjacent(graph, a, d) and \
                   graph_util.adjacent(graph, b, c) and \
                   graph_util.adjacent(graph, d, c) and \
                   graph_util.adjacent(graph, a, c):
                    if graph_util.has_dir_edge(graph, b, c) and \
                       graph_util.has_dir_edge(graph, c, d) and \
                       graph_util.has_undir_edge(graph, a, d):
                        if self.is_arrowpoint_allowed(a, c, knowledge):
                            if not graph_util.is_unshielded_collider(graph, b, a, d):
                                continue

                            if self.is_arrowpoint_allowed(c, d, knowledge):
                                self.direct(c, d, graph)
                                continue
                    e = d
                    d = b
                    b = e
                    if graph_util.has_dir_edge(graph, b, c) and \
                        graph_util.has_dir_edge(graph, c, d) and \
                        graph_util.has_undir_edge(graph, a, d):
                        if self.is_arrowpoint_allowed(a, c, knowledge):
                            if not graph_util.is_unshielded_collider(graph, b, a, d):
                                continue
                            if self.is_arrowpoint_allowed(c, d, knowledge):
                                self.direct(c, d, graph)
                                continue

    def is_arrowpoint_allowed(self, graph, from_node, to_node, knowledge):
        if knowledge is None:
            return True

        return not knowledge.is_required(to_node, from_node) and \
               not knowledge.is_forbidden(from_node, to_node)

    def get_visited(self):
        """ This is what FGES actually uses """
        return self.visited
