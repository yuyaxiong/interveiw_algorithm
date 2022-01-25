# Definition for a binary tree node.

# 222. 完全二叉树的节点个数
# 等比数列公式 Sn = a1 * (1-q**n)/(1-q), q为比值
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def countNodes(self, root: TreeNode) -> int:
        left, right = root, root
        hight_left, hight_right = 0, 0
        while left is not None:
            left = left.left
            hight_left += 1
        while right is not None:
            right = right.right
            hight_right += 1
        if hight_left == hight_right:
            return 2 ** hight_left - 1
        return 1 + self.countNodes(root.left) + self.countNodes(root.right)

class Solution1:
    def countNodes(self, root: TreeNode) -> int:
        if root is None:
            return 0
        if root.left is None and root.right is None:
            return 1
        elif root.left is None and root.right is not None:
            return 1 + self.countNodes(root.right)
        elif root.left is not None and root.right is None:
            return 1 + self.countNodes(root.left)
        elif root.left is not None and root.right is not None:
            return 1 + self.countNodes(root.left) + self.countNodes(root.right)
        

         