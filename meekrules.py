
class MeekRules:
    def __init__(self, undirectUnforcedEdges = True):
        # Unforced parents should be undirected before orienting
        self.undirectUnforcedEdges = undirectUnforcedEdges
        self.nodeSubset = {}

def orientImplied(graph, nodeSubset):
    self.nodeSubset = nodeSubset
    

def orientUsingMeekRulesLocally(knowledge, graph):
    oriented = set({})
