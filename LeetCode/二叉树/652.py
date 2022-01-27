# Definition for a binary tree node.
# 652. 寻找重复的子树
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: Optional[TreeNode]) -> List[Optional[TreeNode]]:
        if root is None:
            return []
        self.subtree_dict = dict()
        self.result = []
        self.traverse(root)
        return self.result

    def traverse(self, root):
        if root is None:
            return "#"
        left, right = self.traverse(root.left), self.traverse(root.right)
        subtree = left + "," + right + "," + str(root.val)
        if self.subtree_dict.get(subtree) is None:
            self.subtree_dict[subtree] = 1
        else:
            self.subtree_dict[subtree] += 1
        if self.subtree_dict.get(subtree) == 2:
            self.result.append(root)
        return subtree