class MeekRules:
    def __init__(self, undirect_unforced_edges = True):
        # Unforced parents should be undirected before orienting
        self.undirect_unforced_edges = undirect_unforced_edges
        self.node_subset = {}

    def orient_implied(self, graph, node_subset):
        self.node_subset = node_subset
        

    def orient_using_meek_rules_locally(self, knowledge, graph):
        oriented = set({})