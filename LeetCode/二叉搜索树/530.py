# Definition for a binary tree node.
#530. 二叉搜索树的最小绝对差
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        pre_node = None
        min_val = 0
        if root is None:
            return min_val
        else:
            min_val = (root.val - root.left.val) if root.left is not None else (root.right.val - root.val)
        min_val, _ = self.getMinDelta(root, min_val, pre_node)
        return min_val

    def getMinDelta(self, root, min_val, pre_node):
        if root is None:
            return min_val, pre_node
        min_val, pre_node = self.getMinDelta(root.left, min_val, pre_node)
        if pre_node is not None:
            min_val = min(root.val - pre_node.val, min_val)
        pre_node = root
        min_val , pre_node = self.getMinDelta(root.right, min_val, pre_node)
        return min_val, pre_node