# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 965. 单值二叉树
class Solution:
    def isUnivalTree(self, root: TreeNode) -> bool:
        if root is None:
            return False
        val = root.val
        return self.univalTree(root, val)

    def univalTree(self, root, val):
        if root is None:
            return True
        if root.val == val:
            return self.univalTree(root.left, val) and self.univalTree(root.right, val)
        else:
            return False