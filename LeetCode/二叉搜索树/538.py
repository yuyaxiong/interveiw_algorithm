# Definition for a binary tree node.
from typing import Optional

# 538. 把二叉搜索树转换为累加树
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if root is None:
            return None
        self.cum_val = 0
        result = 0
        self.CumSumTree(root, result)
        return root

    def CumSumTree(self, root, result):
        if root is None:
            return result
        result = self.CumSumTree(root.right, result)
        result += root.val
        root.val = result
        result = self.CumSumTree(root.left, result)
        return result





