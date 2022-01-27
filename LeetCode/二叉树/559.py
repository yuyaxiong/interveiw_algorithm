"""
# Definition for a Node.
"""
# 559. N 叉树的最大深度
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if root is None:
            return 0
        depth = 0
        for node in root.children:
            depth =  max(self.maxDepth(node), depth)
        return depth + 1


