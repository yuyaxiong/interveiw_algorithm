#700. 二叉搜索树中的搜索

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None:
            return None
        if root.val > val:
            return self.searchBST(root.left, val)
        elif root.val < val:
            return self.searchBST(root.right, val)
        else:
            return root

# 非递归版本
class Solution1:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None:
            return None
        val_node = None
        while root is not None:
            if root.val > val:
                root = root.left 
            elif root.val < val:
                root = root.right 
            else:
                val_node = root
                break
        return val_node

