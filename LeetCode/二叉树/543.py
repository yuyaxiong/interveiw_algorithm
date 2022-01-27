# Definition for a binary tree node.
# 543. 二叉树的直径
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.max_len = 0
        self.getDepth(root)
        return self.max_len

    def getDepth(self, root):
        if root is None:
            return 0
        left_depth, right_depth = self.getDepth(root.left), self.getDepth(root.right)
        self.max_len = max(left_depth + right_depth, self.max_len)
        return max(left_depth, right_depth) + 1 

    