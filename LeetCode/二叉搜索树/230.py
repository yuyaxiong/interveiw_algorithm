# Definition for a binary tree node.
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        k, target = self.getK(root, k, None)
        return target

    def getK(self, root, k, target):
        if root is None:
            return k, target
        k, target = self.getK(root.left, k, target)
        k -= 1
        if k == 0:
            target = root.val
            return k, target
        k, target = self.getK(root.right, k, target)
        return k, target

