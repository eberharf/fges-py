import graph_util

class MeekRules:
    def __init__(self, undirect_unforced_edges = True):
        # Unforced parents should be undirected before orienting
        self.undirect_unforced_edges = undirect_unforced_edges
        self.node_subset = {}
        self.visited = set({})
        self.direct_stack = []

    def orient_implied_subset(self, graph, node_subset):
        self.node_subset = node_subset

    def orient_implied(self, graph):
    	self.orient_implied_subset(graph, graph.nodes())
        

    def orient_using_meek_rules_locally(self, knowledge, graph):
        oriented = set({})

        if (self.undirect_unforced_edges):
        	for node in self.node_subset:
        		self.undirect_unforced_edges(node, graph)
        		self.direct_stack.extend(graph_util.adjacent_nodes(graph, node))

        #TODO: Combine loops
        for node in self.node_subset:
        	self.run_meek_rules(node, graph, knowledge)

        last_node = self.direct_stack.pop()
        while last_node is not None:

        	if self.undirect_unforced_edges:
        		self.undirect_unforced_edges(last_node, graph)

        	self.run_meek_rules(last_node, graph, knowledge)

        	last_node = self.direct_stack.pop()


    def run_meek_rules(self, node, graph, knowledge):
    	self.run_meek_rule_one(node, graph, knowledge)
    	self.run_meek_rule_two(node, graph, knowledge)
    	self.run_meek_rule_three(node, graph, knowledge)
    	self.run_meek_rule_four(node, graph, knowledge)

    def run_meek_rule_one(self, node, graph, knowledge):
    	"""
		Meek's rule R1: if a-->b, b---c, and a not adj to c, then a-->c
		"""
        adjacencies = graph_util.adjacent_nodes(graph, node)
        if len(adjacencies) < 2:
            return 
        all_combinations = itertools.combinations(range(0, len(adjacencies)), 2)
        #TODO: What do a and c represent here?
        for (index_one, index_two) in all_combinations:
            node_a = adjacencies[index_one]
            node_c = adjacencies[index_two]

            #TODO: Parallelize these flipped versions?
            self.r1_helper(node_a, node, node_c, graph, knowledge)
            self.r1_helper(node_c, node, node_a, graph, knowledge)   

    def r1_helper(self, node_a, node_b, node_c, graph, knowledge):
        if (not graph_util.adjacent(g, node_a, node_c) and graph_util.has_dir_edge(g, node_a, node_b) and graph_util.has_undir_edge(graph, node_b, node_c)):
            
            
    
    def run_meek_rule_two(self, node, graph, knowledge):

    def run_meek_rule_two(self, node, graph, knowledge):
        pass
    
    def run_meek_rule_three(self, node, graph, knowledge):
        pass
    
    def run_meek_rule_four(self, node, graph, knowledge):
        pass
		


    def get_visited(self):
    	return self.visited