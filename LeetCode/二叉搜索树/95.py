# Definition for a binary tree node.

# 95. 不同的二叉搜索树 II
from typing import List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def generateTrees(self, n: int) -> List[TreeNode]:
        if n == 0:
            return []
        return self.build(1, n)

    def build(self, left, right):
        res = []
        if left > right:
            res.append(None)
            return res 
        for i in range(left, right+1):
            # 自顶向下        
            left_tree_list = self.build(left, i-1)
            right_tree_list = self.build(i+1, right)
            for left_tree in left_tree_list:
                for right_tree in right_tree_list:
                    node = TreeNode(i)
                    node.left = left_tree
                    node.right = right_tree
                    res.append(node)
        # 自底向上
        return res