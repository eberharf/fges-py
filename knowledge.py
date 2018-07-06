"""
knowledge.py

Description from Tetrad

Stores information about required and forbidden edges and common causes for
use in algorithm. This information can be set edge by edge or else globally
via temporal tiers. When setting temporal tiers, all edges from later tiers
to earlier tiers are forbidden.

For this class, all variable names are referenced by name only. This is
because the same Knowledge object is intended to plug into different graphs
with MyNodes that possibly have the same names. Thus, if the Knowledge object
forbids the edge X --> Y, then it forbids any edge which connects a MyNode
named "X" to a MyNode named "Y", even if the underlying MyNodes themselves
named "X" and "Y", respectively, are not the same.
"""

class Knowledge:
    def __init__(self):
        self.required_edges = {}
        self.forbidden_edges = {}
        self.tier_map = {}
        self.forbidden_within_tiers = {}

    def set_forbidden(self, x, y):
        self.forbidden_edges.add((x, y))

    def remove_forbidden(self, x, y):
        self.forbidden_edges -= {(x, y)}

    def set_required(self, x, y):
        self.required_edges.add((x, y))

    def remove_required(self, x, y):
        self.required_edges -= {(x, y)}

    def set_tier(self, node, tier):
        self.tier_map[node] = tier

    def is_forbidden_by_tiers(self, x, y):
        if self.tier_map[x] == self.tier_map[y]:
            return self.forbidden_within_tiers[x] == self.forbidden_within_tiers[y]
        return self.tier_map[x] < self.tier_map[y]

    def set_tier_forbidden_within(self, tier, forbidden):
        self.forbidden_within_tiers[tier] = forbidden

    def is_forbidden(self, x, y):
        if self.is_forbidden_by_tiers(x, y):
            return True
        else:
            return (x, y) in self.forbidden_edges

    def is_required(self, x, y):
        return (x, y) in self.required_edges