# Definition for a binary tree node.
from typing import Optional

# 450. 删除二叉搜索树中的节点
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if  root is None:
            return None
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if root.left is None or root.right is None:
                root = root.left if root.left is not None else root.right
            else:
                cur = root.right 
                while cur.left is not None:
                    cur = cur.left 
                root.val = cur.val
                root.right = self.deleteNode(root.right, cur.val)
        return root            


