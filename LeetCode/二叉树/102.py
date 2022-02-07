# Definition for a binary tree node.
# 102. 二叉树的层序遍历
from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        if root is None:
            return []
        node_list = [root]
        result = []
        self.recur_line(node_list, result)
        return result

    def recur_line(self, node_list, result):
        if len(node_list) == 0:
            return 
        tmp_list = []
        lines = []
        for node in node_list:
            if node.left is not None:
                tmp_list.append(node.left)
            if node.right is not None:
                tmp_list.append(node.right)
            lines.append(node.val)
        result.append(lines)
        self.recur_line(tmp_list, result)
        
