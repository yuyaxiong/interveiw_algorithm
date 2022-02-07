# Definition for a binary tree node.
from typing import List, Optional

# 144.二叉树的前序遍历
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# 递归版本
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        self.preorder(root, result)
        return result

    def preorder(self, root, result):
        if root is None:
            return 
        result.append(root.val)
        self.preorder(root.left, result)
        self.preorder(root.right, result)

# 非递归版本
class Solution1:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if root is None:
            return []
        node_list = [root]
        result = []
        while len(node_list) > 0: 
            node = node_list.pop(0)
            tmp = []
            result.append(node.val)
            if node.left is not None:
                tmp.append(node.left)
            if node.right is not None:
                tmp.append(node.right)
            node_list = tmp + node_list
        return result 
        

