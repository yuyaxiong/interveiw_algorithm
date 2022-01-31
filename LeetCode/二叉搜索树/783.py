# Definition for a binary tree node.
# 783. 二叉搜索树节点最小距离
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
import sys
class Solution:
    def minDiffInBST(self, root: TreeNode) -> int:
        if root is None:
            return 0
        pre_node = None
        min_val = sys.maxsize
        min_val, pre_node = self.getMin(root, pre_node, min_val)
        return min_val


    def getMin(self, root, pre_node, min_val):
        if root is None:
            return min_val, pre_node
        min_val, pre_node = self.getMin(root.left, pre_node, min_val)
        if pre_node is not None:
            min_val = min(root.val - pre_node.val, min_val)
        pre_node = root
        min_val, pre_node = self.getMin(root.right, pre_node, min_val)
        return min_val, pre_node

