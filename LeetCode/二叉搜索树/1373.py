# Definition for a binary tree node.
# 1373. 二叉搜索子树的最大键值和

from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxSumBST(self, root: Optional[TreeNode]) -> int:
        if root.val is None:
            return 0
        self.max_val = 0
        status, val = self.searchBST(root, True, 0)
        return self.max_val

    def searchBST(self, root, status, val):
        if root is None:
            return True and status, 0
        left_status, left_val = self.searchBST(root.left, status, val)
        right_status, right_val = self.searchBST(root.right, status, val)
        status = left_status and right_status
        if root.left is not None:
            left_node = self.getRightNode(root.left)
            if left_node.val < root.val:
                status = status and True
            else:
                status = status and False
        if root.right is not None:
            right_node = self.getLeftNode(root.right)
            if right_node.val > root.val:
                status = status and True
            else:
                status = status and False
        if status:
            cum_val = root.val + left_val + right_val
            self.max_val = max(cum_val, self.max_val)
        else:
            cum_val = 0
        return status, cum_val

    def getLeftNode(self, root):
        if root is None:
            return None
        while root.left is not None:
            root = root.left
        return root

    def getRightNode(self, root):
        if root is None:
            return None
        while root.right is not None:
            root = root.right
        return root 